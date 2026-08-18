# Copyright (c) 2026, Audio-Technica.
"""量子化しやすい構成部品（このフォーク独自。上流 state-spaces/mamba には無い）。

★このファイルは「上流ファイルへの差分を最小限にする」ために存在する。
  Mamba2 / Mamba3 に足したいクラス・関数はできるだけここに置き、
  上流ファイル側は import 1 行と呼び出し 1 行だけで済ませること
  （上流の更新を取り込むときの競合を避けるため）。

内容:
  - ``DyTGated``     : LayerNorm/RMSNorm の置き換え。除算・平方根を使わない正規化
  - ``dt_activation``: ``dt`` の softplus を ReLU + 下限クリップへ置き換える

背景（Arm Ethos-U65 / Vela の制約）:
  Vela には DIV / SQRT / RSQRT(int16) / SOFTPLUS が無い。RMSNorm は
  ``x * rsqrt(mean(x^2))`` なので、そのままでは NPU に載らないうえ量子化も難しい。
  DyTGated は tanh と MUL/ADD だけで構成されるため、どちらの問題も回避できる。
"""

import torch
import torch.nn.functional as F


class DyTGated(torch.nn.Module):
    """Dynamic Tanh（gated 版）。RMSNormGated / LayerNormGated の差し替え先。

    ``norm(x * silu(z))`` の代わりに ``weight * tanh(alpha * x * silu(z)) + bias`` を計算する。
    統計量（mean / var / rms）を取らないので除算も平方根も現れず、
    出力は tanh により常に ``[-|weight|+bias, |weight|+bias]`` に収まる
    → 量子化レンジが入力に依らず決まるのが利点。

    Args:
        hidden_size: 最終次元のサイズ。``weight`` / ``bias`` の形状になる
        init_alpha:  tanh の入力ゲイン初期値。学習される（スカラー 1 個）
    """

    def __init__(self, hidden_size, init_alpha=0.5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.full((1,), init_alpha, **factory_kwargs))
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size, **factory_kwargs))
        self.alpha._no_weight_decay = True
        self.weight._no_weight_decay = True
        self.bias._no_weight_decay = True

    def forward(self, x, z):
        x = x * F.silu(z)
        x = torch.tanh(self.alpha * x)
        return self.weight * x + self.bias


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
