# Copyright (c) 2026, Audio-Technica.
"""Mamba-3 の逐次実行（1 step）を純 PyTorch で書き下したもの。

上流 ``Mamba3`` は ``forward`` が triton、``step`` が CuteDSL(H100) なので、CUDA が無いと
1 step も動かない。さらに活性化がカーネル内に埋まっており ``mamba3.py`` を読むだけでは見えない
（``pi*tanh(angle)`` は ``ops/triton/mamba3/angle_dt.py``、``sigmoid(trap)`` / RoPE /
台形離散化 / ``silu(z)`` / ``exp(A*dt)`` は ``ops/triton/mamba3/mamba3_siso_fwd.py``）。

このファイルは **量子化側が転写すべき float の関数の正式定義**であり、同時に
**可読性を上げた Mamba-3 アーキテクチャの読み物**でもある。

設計上の約束:

* triton / tilelang / CuteDSL の **演算を一切使わない**。純 PyTorch のみ。
* **実行環境（CUDA / CPU）に依存した分岐を持たない。** ``cuda.is_available()`` /
  ``x.is_cuda`` / ``device.type`` による分岐を書かない。同一コードが両方で走る。
  一方、``is_mimo`` / ``rotate_pairwise`` / ``is_outproj_norm`` の**アーキテクチャの分岐**は
  上流と同じ形で残す（可読性と上流追従のため）。
* **軸のレイアウトを ``Mamba3`` と完全に一致させる。** このファイルの中に軸のアダプタを持たない。
  したがって ``Mamba3`` と ``Mamba3_Step`` は状態をそのまま相互に渡せる。
* ``__init__`` / ``allocate_inference_cache`` は override しない（checkpoint 互換を構造的に保証）。

SISO は「MIMO の R=1・射影が恒等」の特殊ケースとして単一経路で書ける
（``Mamba3.__init__`` が ``is_mimo=False`` のとき ``self.mimo_rank = 1`` に落とすため）。
恒等射影は ``ones`` との積で表現する。IEEE-754 では ``x * 1.0 == x`` が厳密なので、
SISO の数値は分岐を入れた場合と bit 単位で変わらない。
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
from mamba_ssm.modules.quant_friendly import DyT, DyTGated

# 上流が書いた純 PyTorch の RMSNorm。RMSNormGated.forward は triton なので呼べないが、
# 数学はこれと同じ（同じファイルの rmsnorm_fn がこれの triton 版）。自前で書き直さない。
from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref

TWO_PI = 2 * math.pi


def apply_norm(x, norm):
    """非ゲートの norm モジュールを純 PyTorch で適用する（``norm_function`` に対応）。

    ★これは**型（アーキテクチャ）による分岐**で、実行環境による分岐ではない。

    * ``DyT`` は純 PyTorch なので**そのまま呼ぶ**
    * ``RMSNormGated`` は ``forward`` が triton なので呼べない。**構築引数だけ読んで**
      上流の純 PyTorch 実装 ``rms_norm_ref`` に渡す。``RMSNormGated`` は ``weight`` のみを持ち
      ``bias`` は ``None`` 登録（``ops/triton/layernorm_gated.py`` の ``RMSNorm.__init__``）
    """
    if isinstance(norm, DyT):
        return norm(x)
    return rms_norm_ref(
        x,
        norm.weight,
        norm.bias,
        eps=norm.eps,
        group_size=norm.group_size,
        norm_before_gate=norm.norm_before_gate,
    )


def apply_norm_gated(x, z, norm):
    """ゲート付き。``norm_before_gate=True`` なら ``norm(x) * silu(z)``（正規化 → ゲート）。"""
    if isinstance(norm, DyTGated):
        return norm(x, z)
    return rms_norm_ref(
        x,
        norm.weight,
        norm.bias,
        z=z,
        eps=norm.eps,
        group_size=norm.group_size,
        norm_before_gate=norm.norm_before_gate,
    )


def apply_rope(x, cos, sin, rotate_pairwise):
    """RoPE を 1 step ぶん適用する。

    Args:
        x: ``(..., D)``
        cos, sin: ``(..., D_rot // 2)``。``x`` のペア軸 ``D // 2`` に broadcast できる形。
            ``D_rot < D`` のときは ``cos=1`` / ``sin=0`` で ``D // 2`` まで pad する
            （回らない次元は恒等変換になる。上流カーネルが angle を 0 で mask load するのと同じ）。
        rotate_pairwise: ``True`` で **隣接ペア** ``(2i, 2i+1)``（SISO / triton の規約）、
            ``False`` で **半分割ペア** ``(i, i + D/2)``（MIMO / tilelang の規約）。
            上流 ``Mamba3.step`` の ``rotate_pairwise = not is_mimo`` と同じ意味。

    ``x`` は ``expand`` 由来で非 contiguous になり得るので ``view`` ではなく ``reshape`` を使う。
    """
    half = x.shape[-1] // 2
    if cos.shape[-1] < half:
        pad_size = half - cos.shape[-1]
        cos = F.pad(cos, (0, pad_size), value=1.0)
        sin = F.pad(sin, (0, pad_size), value=0.0)

    if rotate_pairwise:
        x_pairs = x.reshape(*x.shape[:-1], half, 2)
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    else:
        x0, x1 = x[..., :half], x[..., half:]

    rotated_0 = x0 * cos - x1 * sin
    rotated_1 = x0 * sin + x1 * cos

    if rotate_pairwise:
        return torch.stack([rotated_0, rotated_1], dim=-1).reshape(x.shape)
    return torch.cat([rotated_0, rotated_1], dim=-1)


def mamba3_recurrence_step(
    q,
    k,
    v,
    adt,
    dt,
    trap,
    angles,
    bias_q,
    bias_k,
    angle_state,
    ssm_state,
    k_state,
    v_state,
    D=None,
    z=None,
    xpj=None,
    zpj=None,
    outpj=None,
    rotate_pairwise=True,
    outproj_norm=None,
):
    """Mamba-3 の状態更新 1 step。**これが float の正式定義**。

    軸は ``Mamba3.allocate_inference_cache`` / ``Mamba3._preprocess`` に一致させている。

    Args:
        q, k: ``(b, R, h, n)`` — ``C_norm(C)`` / ``B_norm(B)``。**bias 加算前・回転前**
        v: ``(b, h, p)`` — ``x``。**``xpj`` 射影前**
        adt: ``(b, h)`` — ``_A * dt``（``_A`` は ``-heavy_tail(dd_A)`` を ``-A_floor`` で clamp）
        dt: ``(b, h)`` — 活性化後の Δ
        trap: ``(b, h)`` — **``sigmoid`` 適用後**
        angles: ``(b, h, Nθ)`` — ``in_proj`` の生値（``tanh`` / ``pi`` は本関数内で掛ける）
        bias_q, bias_k: ``(R, h, n)`` — ``C_bias`` / ``B_bias`` を ``h r n -> r h n`` にしたもの
        angle_state: ``(b, h, Nθ)`` / ssm_state: ``(b, h, p, n)``
        k_state: ``(b, R, h, n)`` / v_state: ``(b, h, p)``（**射影前の生の v**）
        xpj, zpj, outpj: ``(R, h, p)`` — ``mimo_x`` / ``mimo_z`` / ``mimo_o``。SISO では ``ones``
        outproj_norm: ``is_outproj_norm=True`` のときの出力 norm モジュール。``None`` なら
            ``silu(z)`` のゲートのみ

    Returns:
        y: ``(b, h, p)`` — ``out_proj`` 適用前
        angle_state, ssm_state, k_state, v_state: 次 step 用（入力と同じ形）
    """
    # ---- 累積角の更新（π·tanh はカーネル内の活性化。angle_dt.py L94 / 参照実装 L62 と同順） ----
    angles = torch.tanh(angles.float()) * math.pi
    angle_state = angle_state + angles * dt.float().unsqueeze(-1)
    # mod 2π。cos/sin は 2π 周期なので数学的には省略可能だが、上流 angle_dt.py L108 と
    # SISO 参照実装 L111 が適用するので合わせる。
    angle_state = angle_state - TWO_PI * torch.floor(angle_state / TWO_PI)

    cos_angles = torch.cos(angle_state).unsqueeze(1)  # (b, 1, h, Nθ)
    sin_angles = torch.sin(angle_state).unsqueeze(1)

    # ---- bias は回転より前に足す（mamba3_siso_fwd.py L313-316 / 参照実装 L100-101） ----
    q = q + bias_q.unsqueeze(0)
    k = k + bias_k.unsqueeze(0)
    q_rot = apply_rope(q, cos_angles, sin_angles, rotate_pairwise)
    k_rot = apply_rope(k, cos_angles, sin_angles, rotate_pairwise)

    # ---- 指数台形離散化。beta に alpha が掛かるのが「台形」の本質 ----
    alpha = torch.exp(adt)
    beta = (1 - trap) * dt * alpha
    gamma = trap * dt

    # ---- v を rank 軸へ射影する。射影は (R, h, p) の要素積で時間不変なので、
    #      状態としては射影前の v を 1 本持てば足りる（= v_state に rank 軸が無い理由）。 ----
    v_proj = v.unsqueeze(1) * xpj.unsqueeze(0)                # (b, R, h, p)
    v_state_proj = v_state.unsqueeze(1) * xpj.unsqueeze(0)    # (b, R, h, p)

    # ---- 状態更新。R=1 で参照実装の式に厳密に縮退する形で書く（einsum は縮約経路が変わる） ----
    ssm_state = alpha.unsqueeze(-1).unsqueeze(-1) * ssm_state
    ssm_state = ssm_state + beta.unsqueeze(-1).unsqueeze(-1) * (
        v_state_proj.unsqueeze(-1) * k_state.unsqueeze(-2)
    ).sum(dim=1)
    ssm_state = ssm_state + gamma.unsqueeze(-1).unsqueeze(-1) * (
        v_proj.unsqueeze(-1) * k_rot.unsqueeze(-2)
    ).sum(dim=1)

    # ---- 出力。(b,1,h,p,n) @ (b,R,h,n,1) -> (b,R,h,p,1) ----
    y = (
        ssm_state.unsqueeze(1) @ q_rot.to(ssm_state.dtype).unsqueeze(-1)
    ).squeeze(-1)  # (b, R, h, p)

    if D is not None:
        y = y + D[None, None, :, None] * v_proj

    if outproj_norm is None:
        # ゲートのみ。参照実装 L136 と同じ順序（out * z * sigmoid(z)）。
        if z is not None:
            z_proj = z.unsqueeze(1) * zpj.unsqueeze(0)  # (b, R, h, p)
            y = y * z_proj * torch.sigmoid(z_proj)
        y = (y * outpj.unsqueeze(0)).sum(dim=1)  # (b, h, p)
    else:
        # 正規化 → ゲート（norm_before_gate=True）。Mamba3._postprocess と同じ順序。
        headdim = y.shape[-1]
        z_proj = torch.einsum("bhp,rhp->brhp", z.float(), zpj)
        z_proj = rearrange(z_proj, "b r h p -> b r (h p)")
        y = rearrange(y, "b r h p -> b r (h p)").float()
        y = apply_norm_gated(y, z_proj, outproj_norm)
        y = rearrange(y, "b r (h p) -> b r h p", p=headdim)
        y = torch.einsum("brhp,rhp->bhp", y, outpj)

    return y, angle_state, ssm_state, k_rot, v


class Mamba3_Step(Mamba3):
    """``Mamba3`` の純 PyTorch 逐次実行版。

    ``__init__`` と ``allocate_inference_cache`` は継承のみ（override しない）ので、
    ``state_dict`` のキー・形状と状態テンソルの形は ``Mamba3`` と構造的に一致する。

    ``B_norm`` / ``C_norm`` / ``norm`` は ``RMSNormGated``（triton）のまま保持するが
    **呼ばない**。構築引数（``weight`` / ``eps`` / ``group_size`` / ``norm_before_gate``）を
    読んで上流の純 PyTorch 実装 ``rms_norm_ref`` に渡す。

    ``_preprocess`` / ``_postprocess`` は再利用しない。前者は triton norm を呼ぶうえ
    ``trap`` に ``sigmoid`` を掛けるので（``forward`` 経路はカーネル内で掛ける）
    二重適用の罠がある。
    """

    def _mimo_projections(self, device, dtype):
        """``mimo_x`` / ``mimo_z`` / ``mimo_o`` を ``(R, h, p)`` で返す。

        SISO では ``Mamba3.__init__`` がこれらを作らないので、上流 ``Mamba3.step``
        L378-380 と同じく ``ones``（恒等射影）を作る。
        """
        if self.is_mimo:
            xpj = rearrange(self.mimo_x, "h r p -> r h p").contiguous()
            zpj = rearrange(self.mimo_z, "h r p -> r h p").contiguous()
            outpj = rearrange(self.mimo_o, "h r p -> r h p").contiguous()
        else:
            shape = (self.mimo_rank, self.nheads, self.headdim)
            xpj = torch.ones(shape, device=device, dtype=dtype)
            zpj = torch.ones(shape, device=device, dtype=dtype)
            outpj = torch.ones(shape, device=device, dtype=dtype)
        return xpj, zpj, outpj

    def _split_in_proj(self, u):
        """``in_proj`` の出力を上流と同じ順・同じ幅で分割し、同じレイアウトに整える。

        順序は ``Mamba3.__init__`` L106 のコメント ``[z, x, B, C, dd_dt, dd_A, trap, angle]``。

        Args:
            u: ``(b, d_model)``
        Returns:
            q, k: ``(b, R, h, n)``（norm 済み・bias 前） / v, z: ``(b, h, p)``
            adt, dt, trap: ``(b, h)``（``trap`` は sigmoid 済み） / angles: ``(b, h, Nθ)``（生値）
        """
        zxbcdt = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        _A = -heavy_tail_activation(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        adt = _A * dt
        trap = torch.sigmoid(trap)

        B = rearrange(B, "b (r g n) -> b r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b (r g n) -> b r g n", r=self.mimo_rank, g=self.num_bc_heads)
        B = apply_norm(B, self.B_norm)
        C = apply_norm(C, self.C_norm)
        # ngroups=1 では B/C は全ヘッド共有。ヘッド差は B_bias / C_bias のみ。
        B = B.expand(-1, -1, self.nheads, -1)
        C = C.expand(-1, -1, self.nheads, -1)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
        # 回転角は全ヘッド共通（in_proj は num_rope_angles 個しか出さない）。
        angles = angles.unsqueeze(-2).expand(-1, self.nheads, -1)

        return C, B, x, z, adt, dt, trap, angles

    def _step_core(self, u, angle_state, ssm_state, k_state, v_state):
        """1 step ぶん。``out_proj`` は掛けず ``(b, h, p)`` を返す。

        ``forward`` と ``step`` の共通部分。参照実装 ``mamba3_siso_step_ref`` /
        ``mamba3_MIMO_step_ref`` の戻り値も ``out_proj`` 前なので、検証はここに当てる。
        """
        q, k, v, z, adt, dt, trap, angles = self._split_in_proj(u)
        xpj, zpj, outpj = self._mimo_projections(v.device, v.dtype)

        return mamba3_recurrence_step(
            q=q,
            k=k,
            v=v,
            adt=adt,
            dt=dt,
            trap=trap,
            angles=angles,
            bias_q=rearrange(self.C_bias, "h r n -> r h n"),
            bias_k=rearrange(self.B_bias, "h r n -> r h n"),
            angle_state=angle_state,
            ssm_state=ssm_state,
            k_state=k_state,
            v_state=v_state,
            D=self.D,
            z=z,
            xpj=xpj,
            zpj=zpj,
            outpj=outpj,
            # MIMO/tilelang は回転行列を置換して i と i+N/2 をペアにする（Mamba3.step L360-363）。
            rotate_pairwise=not self.is_mimo,
            outproj_norm=self.norm if self.is_outproj_norm else None,
        )

    def step(self, u, angle_state, ssm_state, k_state, v_state, **kwargs):
        """``Mamba3.step`` と同じシグネチャ・同じ戻り値。状態は in-place で更新する。

        Args:
            u: ``(batch, d_model)``
            angle_state: ``(batch, nheads, num_rope_angles)``
            ssm_state: ``(batch, nheads, headdim, d_state)``
            k_state: ``(batch, R, nheads, d_state)``, R = ``mimo_rank``（非 MIMO では 1）
            v_state: ``(batch, nheads, headdim)``
        Returns:
            out, nxt_angle_state, ssm_state, nxt_k_state, nxt_v_state
        """
        y, nxt_angle_state, nxt_ssm_state, nxt_k_state, nxt_v_state = self._step_core(
            u, angle_state, ssm_state, k_state, v_state
        )

        out = rearrange(y, "b h p -> b (h p)")
        out = self.out_proj(out.to(self.out_proj.weight.dtype))

        angle_state.copy_(nxt_angle_state)
        ssm_state.copy_(nxt_ssm_state)
        k_state.copy_(nxt_k_state)
        v_state.copy_(nxt_v_state)

        return out, nxt_angle_state, ssm_state, nxt_k_state, nxt_v_state

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        """``Mamba3.forward`` と同じシグネチャ。中身は ``step`` の逐次ループ。

        ``out_proj`` は時間ごとに独立な線形写像なので、逐次に掛けても並列に掛けても同じ。

        varlen（``cu_seqlens``）は非対応（【これからやること】1 で「不要」と判断済み）。
        """
        assert cu_seqlens is None, "Mamba3_Step does not support varlen (cu_seqlens)."
        batch, seqlen, _ = u.shape

        cache = None
        if inference_params is not None:
            cache = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # decode。``step`` は ``(b, d_model)`` を取るので長さ軸を落とす。
                # ★上流 Mamba3.forward L172 は ``(b, l, d)`` をそのまま step へ渡しており、
                #   ``_preprocess`` の rearrange が 3 次元入力で落ちる。上流ファイルは
                #   触らず、こちら側で軸を合わせて decode が実際に動くようにする。
                #   戻り値の rank は入力に合わせる（上流は rank 2 を返して不整合）。
                assert seqlen == 1, "decode (seqlen_offset > 0) supports seqlen=1 only."
                out, _, _, _, _ = self.step(u[:, 0], *cache)
                return out.unsqueeze(1)
            states = cache
        else:
            # 状態の形は Mamba3.allocate_inference_cache に完全一致する（override していない）。
            states = self.allocate_inference_cache(
                batch, seqlen, device=u.device, dtype=u.dtype
            )

        angle_state, ssm_state, k_state, v_state = states
        ys = []
        for idx in range(seqlen):
            y, angle_state, ssm_state, k_state, v_state = self._step_core(
                u[:, idx], angle_state, ssm_state, k_state, v_state
            )
            ys.append(y)

        if cache is not None:
            # ★最終状態を cache に書き戻す。上流 Mamba3.forward L266-271 と同じ責務。
            # これを忘れると、続く decode（seqlen_offset > 0）がゼロ状態から再開してしまう。
            for buf, val in zip(cache, (angle_state, ssm_state, k_state, v_state)):
                buf.copy_(val)

        y = torch.stack(ys, dim=1)  # (b, l, h, p)
        out = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(out.to(self.out_proj.weight.dtype))
