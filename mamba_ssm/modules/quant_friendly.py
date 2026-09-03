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

import math

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


# ---------------------------------------------------------------------------
# Mamba-3 の dt パラメータ化（``dt_transform``）
#
# ★順変換と逆変換を必ず**対で**ここに置くこと。「順変換と逆変換の取り違え」が
#   この機能の唯一のバグ class であり、離れた場所に書くと必ず食い違う。
#   互いの逆であることは verify/verify_mamba3_dt.py が数値で検査する。
#
# ★論文の仕様は「``dt_bias`` の値」ではなく「**活性化後の Δ の範囲**」で決まっている
#   （Mamba 論文: Δ は ``τ_Δ⁻¹(Uniform([dt_min, dt_max]))`` で初期化。``τ_Δ`` は使う活性化）。
#   したがって ``inv_dt_transform`` は「その活性化の逆関数を当てているだけ」で、
#   inverse-softplus は softplus 固有の仕様ではない。
#
# ★Mamba-2 用の ``dt_activation``（上）とは**意図的に分けている**。
#   Mamba-2 は softplus/relu の切り替えで、カーネル側（ssd_combined.py の ``dt_softplus``）にも
#   同じ分岐が入っている。Mamba-3 のカーネルは dt に活性化を当てないので事情が違う。
#   softplus の記述が 2 箇所に出る軽い重複は、**Mamba-2 の bit 一致を守る対価**として受け入れる。
# ---------------------------------------------------------------------------

DT_TRANSFORMS = ("softplus", "exp")


def _check_dt_transform(dt_transform):
    if dt_transform in DT_TRANSFORMS:
        return
    if dt_transform == "relu":
        raise ValueError(
            "dt_transform='relu' is not supported for Mamba-3. In Mamba-3, Δ also scales the "
            "rotation angle (Δθ = π·tanh(angle)·Δ), so losing softplus's multiplicative "
            "compression is not acceptable: the local gain ratio (dΔ/du)/Δ becomes 1/Δ "
            "(100 at Δ=0.01, 1000 at Δ=0.001) instead of ~1, which spreads Δ over 4.25 decades "
            "and pushes ~30% of Δ above 1/π = 0.318 where the deploy-side Taylor cos/sin breaks. "
            "Mamba-2 uses relu (see dt_activation) because it has no rotation path."
        )
    raise ValueError(
        f"Unknown dt_transform {dt_transform!r}. Expected one of {DT_TRANSFORMS}."
    )


def apply_dt_transform(x, dt_transform):
    """事前活性化 ``x = dd_dt + dt_bias`` から Δ を作る（順変換 ``τ_Δ``）。

    * ``"softplus"``: 上流の既定。``F.softplus(x)``
    * ``"exp"``: ``torch.exp(x)``。**Vela は SOFTPLUS/LOG 非対応だが EXP は int8/int16 で対応**。
      softplus が動作域で担っていた**乗法的圧縮**を保ち（局所ゲイン比が厳密に 1）、
      正値性は softplus より強く厳密に > 0（下限クリップ不要）。
      仕様帯 Δ∈[0.001, 0.1] では softplus との差が 0.05〜5% しかない
    """
    _check_dt_transform(dt_transform)
    if dt_transform == "softplus":
        return F.softplus(x)
    return torch.exp(x)


def inv_dt_transform(dt, dt_transform):
    """Δ から事前活性化を逆算する（逆変換 ``τ_Δ⁻¹``）。``dt_bias`` の初期化に使う。

    ``apply_dt_transform(inv_dt_transform(dt, t), t) == dt`` が成り立つこと（検証済み）。

    * ``"softplus"``: ``dt + log(-expm1(-dt))``。★上流 ``mamba3.py`` の式と同一
    * ``"exp"``: ``log(dt)``。★対数一様初期化は元々 exp 用に書かれている
      （``_dt = exp(U·(log dt_max − log dt_min) + log dt_min)`` なので ``log(_dt)`` は指数部そのもの）
    """
    _check_dt_transform(dt_transform)
    if dt_transform == "softplus":
        return dt + torch.log(-torch.expm1(-dt))
    return torch.log(dt)


# ---------------------------------------------------------------------------
# Δ の上限（``dt_limit``）
#
# ★なぜ要るか: Mamba-3 では Δ が**回転角にも掛かる**（``Δθ = π·tanh(angle)·Δ``）。
#   デプロイ側は COS/SIN が無く Taylor 多項式で cos/sin を作るので定義域は |Δθ| <= 1 rad。
#   ``π·tanh`` で角度側は ±π に有界なので **Δ <= 1/π を保証すれば足りる**。
#   累積角 Θ の 360° 超えは問題ない（trig_state 設計ではラップアラウンドが原理的に不要）。
#   厳密なのは 1 step の Δθ だけ。
#
# ★引数名・形は**上流 Mamba-2 の ``dt_limit=(0.0, float("inf"))`` と同じ**。新しい名前を発明しない。
#
# ★hard clamp を採った理由（2026-08-21、3 候補を実測して決定）:
#   * 仕様帯 [0.001, 0.1] で**厳密に恒等** ＝ ``dt_transform`` の結論（Δ の分布・初期化仕様・
#     乗法的圧縮）を bit 単位で保存する。初期化の逆変換も変更不要（``log Δ₀`` のまま）
#   * 到達可能域の局所ゲイン比が**厳密に 1.0**（tanh 版は Δ=0.3 で 0.21、σ 版は 0.06 まで落ちる）
#   * 唯一の短所「上限域で勾配が 0」は **上流 Mamba-2 が既に採用・出荷している挙動**
#     （``ops/triton/ssd_chunk_state.py`` の ``clamp_mask`` → ``ddt = tl.where(clamp_mask, 0.0, ddt)``
#     が ``torch.clamp`` と同じ勾配）。Δmax=1/π では**該当が 0.195% だけ**
#   * デプロイ側は ``Δmax − relu(Δmax − Δ)`` で書ける（RELU は前段の fused activation に
#     畳まれるので SUB 1 個ぶんの追加で済む）。★この relu の綴りは **TF 側の仕事**であり、
#     PyTorch 側は ``torch.clamp`` と書く（実機モデルが ``torch.clamp`` を
#     ``relu(x + 16.0) - 16.0`` として書いているのと同じ関係）
# ---------------------------------------------------------------------------

#: Taylor cos/sin の定義域 |Δθ| <= 1 rad から出る Δ の上限。``Δθ = π·tanh(angle)·Δ`` で
#: 角度側が ±π に有界なので、``Δ <= 1/π`` なら |Δθ| <= 1 rad が保証される。
DT_MAX_TAYLOR = 1.0 / math.pi  # 0.3183...

NO_DT_LIMIT = (0.0, float("inf"))


def apply_dt_limit(dt, dt_limit=NO_DT_LIMIT):
    """Δ に上下限を掛ける。上流 Mamba-2 の ``dt_limit`` と同じ数学・同じ勾配（hard clamp）。

    ★既定 ``(0.0, inf)`` は**恒等**（早期 return するので bit 単位で無変更）。
    上流一致テストの錨を守るため、既定で 1 bit も変えないことが重要。

    Args:
        dt: 活性化後の Δ（``apply_dt_transform`` の出力）
        dt_limit: ``(dt_min, dt_max)``。``Δ <= 1/π`` を保証したいなら
            ``(0.0, DT_MAX_TAYLOR)`` を渡す
    """
    dt_min, dt_max = dt_limit
    if dt_min <= 0.0 and dt_max == float("inf"):
        return dt
    return dt.clamp(min=dt_min, max=dt_max)


# ---------------------------------------------------------------------------
# Mamba-2 の A の下限（``A_floor``）
#
# ★上流は ``A = -exp(A_log)`` なので |A| に下限が無く、学習で ``A_log -> -inf`` に
#   落ちると ``dA = exp(dt·A) -> 1`` になる（減衰が消える）。
#
# ★★**なぜ dt ではなく A に下限を置くのか**:
#   このフォークの Mamba-2 は ``softplus_to_relu=True`` のとき
#   ``dt_act = relu(dt + dt_bias) + DT_FLOOR`` で **dt に構造的な下限 1e-4** がある
#   （``dt_activation`` / ``ssd_chunk_state.py`` / ``ssd_combined.py`` の 3 経路すべて）。
#   したがって |A| に下限を置くだけで **dA の上界が構成として確定する**:
#
#       dA = exp(-dt_act·|A|) <= exp(-DT_FLOOR · A_floor)
#
#   | A_floor | 保証される dA の上限 |
#   |---------|---------------------|
#   | 100.5   | 0.99                |
#   | 202     | 0.98                |
#   | 513     | 0.95                |
#
#   ★これが効くのは、状態が毎フレーム丸められその誤差が極の近さに応じて積み上がる
#   （蓄積利得 ``1/(1-dA²)``）ため。dA <= 0.99 で上限が 17.0 dB に固定される。
#   ★逆に **dt に一律の下限**を置くと、|A| が大きいヘッドが ``dA = exp(-21) ~ 0`` になり
#   **既に記憶が短くて無害なヘッドの再帰を壊す**。A 側に置くのが形として正しい。
#
# ★★**保証が成立する条件**: ``softplus_to_relu=True``。既定の softplus 経路には
#   dt の床が無い（``softplus(dt) -> 0``）ので、上界は成立しない。
#
# ★clamp ではなく**再パラメータ化**を使う。Mamba-3 の ``A_floor`` は A が
#   データ依存の活性化なので ``clamp(_A, max=-A_floor)`` しか選べず上限域で勾配が死ぬが、
#   Mamba-2 の A は学習パラメータなので ``-(A_floor + exp(A_log))`` と書けて
#   **全域で滑らか・勾配が死なない**。
#
# ★初期化は変更しない。``A_floor`` は ``A_init_range`` より桁で大きいのが普通なので
#   （既定 ``(1, 16)`` に対し 100.5）、逆変換 ``log(|A| - A_floor)`` は成立しない。
#   ``A_init_range`` は「**床からの上乗せ幅**」と解釈する（|A| = A_floor + U(A_init_range)）。
# ---------------------------------------------------------------------------

A_NO_FLOOR = 0.0


def apply_A_floor(A_log, A_floor=A_NO_FLOOR):
    """``A = -(A_floor + exp(A_log))``。|A| >= A_floor が恒等的に成立する。

    ★既定 ``0.0`` は**恒等**（早期 return するので bit 単位で無変更）。
    上流一致・既存 checkpoint との bit 一致の錨を守るため、既定で 1 bit も変えない。

    Args:
        A_log: ``self.A_log.float()``
        A_floor: |A| の下限。``exp(-DT_FLOOR * A_floor)`` が dA の上界になる
    """
    if A_floor == A_NO_FLOOR:
        return -torch.exp(A_log)
    return -(A_floor + torch.exp(A_log))
