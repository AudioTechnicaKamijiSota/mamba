# Copyright (c) 2026, Audio-Technica.
"""量子化しやすい構成部品（このフォーク独自。上流 state-spaces/mamba には無い）。

★このファイルは「上流ファイルへの差分を最小限にする」ために存在する。
  Mamba2 / Mamba3 に足したいクラス・関数はできるだけここに置き、
  上流ファイル側は import 1 行と呼び出し 1 行だけで済ませること
  （上流の更新を取り込むときの競合を避けるため）。

内容:
  - ``DyT``          : RMSNorm（非ゲート）の置き換え。除算・平方根を使わない正規化
  - ``DyTGated``     : LayerNormGated/RMSNormGated の置き換え
  - ``build_norm``   : ``norm_function`` から norm モジュールを組むファクトリ
  - ``dt_activation``: ``dt`` の softplus を ReLU + 下限クリップへ置き換える

背景（Arm Ethos-U65 / Vela の制約）:
  Vela には DIV / SQRT / RSQRT(int16) / SOFTPLUS が無い。RMSNorm は
  ``x * rsqrt(mean(x^2))`` なので、そのままでは NPU に載らないうえ量子化も難しい。
  DyT は tanh と MUL だけで構成されるため、どちらの問題も回避できる。

★**数式としての置き換え対応**（各演算の目的をずらさないこと）:

  RMSNorm : out = weight * x / sqrt(mean(x^2) + eps)
                    |         |
                    |         +-- tanh(alpha * x) で置き換える（統計量で割る演算）
                    +------------ そのまま残す（チャネル毎スケール）

  DyT     : out = weight * tanh(alpha * x)

  → 新しく増えるパラメータは ``alpha`` の 1 個だけ。

★**bias（DyT 論文の beta）は既定で持たない。** 論文の DyT は LayerNorm
  （``gamma * (x - mu) / sigma + beta``）を対象にしているが、**上流が使っているのは RMSNorm で
  bias を持たない**（``layernorm_gated.py`` の ``RMSNorm.__init__`` が
  ``register_parameter("bias", None)``）。外側の演算まで見ると:

  * ``B_norm`` / ``C_norm``: 直後に ``q = C_norm(C) + C_bias`` があり、``C_bias`` は
    ``(nheads, R, d_state)`` で**共有 beta より表現力が上**。beta は ``C_bias`` に厳密に
    吸収されるので、足すと**既にある外側の演算と重複**する
  * 出力 norm: ``out_proj`` が ``bias=False`` なので外側に相当物が無い。足すと
    **上流にもその外側にも無い能力の新規追加**になる（``norm_before_gate=True`` では
    beta が ``* silu(z)`` で変調され「ゲートで変調される定数」という上流に無い項になる）

  → どちらも beta を足さない。``bias=True`` で明示的に有効化はできる
  （Mamba-2 側の既存 ``DyTGated`` が bias を持っているため、後方互換のために残してある）。
"""

import torch
import torch.nn.functional as F

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated


class DyT(torch.nn.Module):
    """Dynamic Tanh（非ゲート版）。``RMSNormGated`` を z なしで使っている箇所の差し替え先。

    ``weight * tanh(alpha * x)`` を計算する。統計量（mean / var / rms）を取らないので
    除算も平方根も現れず、出力は tanh により常に ``[-|weight|, |weight|]`` に収まる
    → 量子化レンジが入力に依らず決まり、対称量子化と相性が良いのが利点。

    ★**失われる性質**: ``/rms(x)`` が与えていた**入力スケール不変性**は無くなる
    （``tanh`` は大きさに敏感で、大きい入力は飽和で潰れる）。仕様変更として扱う。

    Args:
        hidden_size: 最終次元のサイズ。``weight`` の形状になる
        init_alpha:  tanh の入力ゲイン初期値。学習される（スカラー 1 個）。
            既定 0.5 は arXiv:2503.10622 が非 LLM の既定として「0.5〜1.2 で安定」と述べている値
        bias: ``True`` で DyT 論文の beta を持つ。★既定 False（上記の理由）
    """

    def __init__(self, hidden_size, init_alpha=0.5, bias=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.full((1,), init_alpha, **factory_kwargs))
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.alpha._no_weight_decay = True
        self.weight._no_weight_decay = True
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size, **factory_kwargs))
            self.bias._no_weight_decay = True
        else:
            # RMSNormGated と同じく bias 無しを None で表す（state_dict にも現れない）
            self.register_parameter("bias", None)

    def forward(self, x, z=None):
        """``RMSNormGated.forward(x, z=None)`` と同じシグネチャ（drop-in にするため）。"""
        assert z is None, "DyT is not gated. Use DyTGated for the gated variant."
        out = self.weight * torch.tanh(self.alpha * x)
        return out if self.bias is None else out + self.bias


class DyTGated(torch.nn.Module):
    """Dynamic Tanh（gated 版）。RMSNormGated / LayerNormGated の差し替え先。

    ``norm_before_gate`` でゲートと正規化の順序を選ぶ（``RMSNormGated`` と同じ意味）:

    * ``False``（既定）: ``weight * tanh(alpha * x * silu(z)) + bias``（ゲート → 正規化）
    * ``True``        : ``(weight * tanh(alpha * x) + bias) * silu(z)``（正規化 → ゲート）

    ★**既定が False なのは Mamba-2 の挙動を 1 bit も変えないため。**
    Mamba-2 側は引数を渡さずに構築しており、``epoch_138`` の実重みで ``kamiji_dev1`` と
    bit 一致している到達点がある。Mamba-3 の出力 norm は上流が
    ``norm_before_gate=True`` で構築するので、そちらは明示的に True を渡す。

    Args:
        hidden_size: 最終次元のサイズ。``weight`` / ``bias`` の形状になる
        init_alpha:  tanh の入力ゲイン初期値。学習される（スカラー 1 個）
        norm_before_gate: 上記のとおり。``RMSNormGated`` の同名引数と同じ意味
        bias: ``True``（既定）で DyT 論文の beta を持つ。★Mamba-3 からは False で呼ぶ
            （理由はモジュール docstring）
    """

    def __init__(self, hidden_size, init_alpha=0.5, norm_before_gate=False, bias=True,
                 device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm_before_gate = norm_before_gate
        self.alpha = torch.nn.Parameter(torch.full((1,), init_alpha, **factory_kwargs))
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.alpha._no_weight_decay = True
        self.weight._no_weight_decay = True
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size, **factory_kwargs))
            self.bias._no_weight_decay = True
        else:
            self.register_parameter("bias", None)

    def forward(self, x, z):
        if not self.norm_before_gate:
            x = x * F.silu(z)
        x = torch.tanh(self.alpha * x)
        out = self.weight * x if self.bias is None else self.weight * x + self.bias
        if self.norm_before_gate:
            out = out * F.silu(z)
        return out


def build_norm(norm_function, hidden_size, *, eps=1e-5, gated=False, norm_before_gate=True,
               group_size=None, bias=False, init_alpha=0.5, device=None, dtype=None):
    """``norm_function`` から norm モジュールを組む。

    ★このファクトリの目的は**上流ファイルへの差分を最小にすること**。
    ``mamba3.py`` 側は構築 1 行をこの呼び出しに置き換えるだけで済み、行数もインデントも変わらない。

    ★``norm_function="RMSNorm"`` は ``RMSNormGated`` を**上流と厳密に同一の引数**で返す。
    これが上流一致テストの唯一の錨になるので、ここで既定値を変えてはいけない。

    Args:
        norm_function: ``"RMSNorm"``（上流）/ ``"DyT"`` / ``"DyTGated"``（後 2 つは同義。
            既存プロジェクトが ``"DyTGated"`` の名前で渡すため両方受ける）
        gated: ``True`` で ``forward(x, z)`` を取るゲート付き版を返す
        norm_before_gate / group_size / eps: ``RMSNormGated`` にそのまま渡す。
            ★``group_size`` は DyT では意味を失う（統計量を取らないため）が、
            **RMSNorm に戻したときに同一になるよう引数は受け続ける**
        bias: DyT の beta。既定 False（モジュール docstring の理由）
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    if norm_function == "RMSNorm":
        return RMSNormGated(hidden_size, eps=eps, group_size=group_size,
                            norm_before_gate=norm_before_gate, **factory_kwargs)
    if norm_function in ("DyT", "DyTGated"):
        if gated:
            return DyTGated(hidden_size, init_alpha=init_alpha,
                            norm_before_gate=norm_before_gate, bias=bias, **factory_kwargs)
        return DyT(hidden_size, init_alpha=init_alpha, bias=bias, **factory_kwargs)
    raise ValueError(
        f"Unknown norm_function {norm_function!r}. Expected 'RMSNorm', 'DyT', or 'DyTGated'. "
        "('LayerNorm' is supported by Mamba2 only; it is triton-based and has no pure-PyTorch "
        "path in mamba3_step.py.)"
    )


# ★dt の下限。softplus の出力は常に正だが、ReLU に置き換えると 0 になり得る。
#   dA = exp(dt * A) が dt=0 で 1 になって状態が減衰しなくなるのを防ぐための床。
#   値は Mamba2 の dt_init_floor と同じ 1e-4。
#   ★triton カーネル側（ssd_chunk_state.py / selective_state_update.py）にも
#     同じ定数がリテラルで入っている。変えるときは 3 箇所すべて合わせること。
DT_FLOOR = 1e-4


def dt_activation(dt, dt_softplus, dt_floor=DT_FLOOR):
    """``dt`` を非負にする活性化。``dt_softplus=False`` で ReLU + 下限クリップになる。

    softplus は Ethos-U に対応する op が無く、量子化も難しい（入力レンジが広く、
    出力が 0 付近で急に潰れる）。ReLU + 定数加算は MUL/ADD/RELU だけで表せる。

    ★この式は triton カーネル側の ``else: dt = tl.maximum(dt, 0.0) + 1e-4`` と
      一致していること（parallel 経路と streaming 経路で dt がずれると
      学習と推論で挙動が変わる）。
    """
    if dt_softplus:
        return F.softplus(dt)
    return F.relu(dt) + dt_floor
