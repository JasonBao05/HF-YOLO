"""
HF-YOLO: Frequency-Aware Linear Attention (FALA) Module
=================================================================
6 sub modules:
  S1: Focused kernel φ_p         — sharpen attention distribution
  S2: WTConv for LePE            — wavelet frequency decomposition
  S3: 2D RoPE-Mixed              — per-head multi-frequency encoding
  S4: SwiGLU FFN                 — adaptive gating, saves 5% GFLOPs
  A6: RepDWConv                  — mid-frequency context (free at inference)
  A8: Learnable scaled residual  — gradient path (inner + outer)

Naming:
  FALABlock   — core attention block (replaces MLLABlock)
  C3kFALA     — C3k with FALABlock inside (replaces C3kMLLABlock)
  CSP_FALA    — full CSP integration (replaces C3k2_MLLABlock2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CSP_FALA']


# ================================================================
#  Utilities
# ================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ================================================================
#  S1: Focused Kernel
# ================================================================

def focused_kernel(x, p=3):
    """φ(x) = |x|^p / ||x||^p — sharpens attention distribution."""
    x_abs = x.abs() + 1e-6
    x_pow = x_abs ** p
    return x_pow / (x_pow.norm(dim=-1, keepdim=True) + 1e-6)


# ================================================================
#  S2: WTConv — wavelet LePE
# ================================================================

class WTConv2d(nn.Module):
    """Haar wavelet convolution: decomposes into LL/LH/HL/HH subbands,
    applies separate DWConv on each, then reconstructs."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_ll = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv_hf = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def _dwt2d(self, x):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def _idwt2d(self, ll, lh, hl, hh, H_orig, W_orig):
        B, C, H2, W2 = ll.shape
        out = ll.new_zeros(B, C, H2 * 2, W2 * 2)
        out[:, :, 0::2, 0::2] = (ll - lh - hl + hh) * 0.5
        out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        out[:, :, 1::2, 1::2] = (ll + lh + hl + hh) * 0.5
        return out[:, :, :H_orig, :W_orig]

    def forward(self, x):
        B, C, H, W = x.shape
        if H < 4 or W < 4:
            return self.bn(self.conv_ll(x))
        ll, lh, hl, hh = self._dwt2d(x)
        ll = self.conv_ll(ll)
        hf = torch.cat([lh, hl, hh], dim=1)
        hf = self.conv_hf(hf)
        lh, hl, hh = hf.chunk(3, dim=1)
        return self.bn(self._idwt2d(ll, lh, hl, hh, H, W))


# ================================================================
#  S3: 2D RoPE-Mixed — pure real-number, NO complex ops
# ================================================================

class RoPE2DMixed(nn.Module):
    """2D RoPE with per-head learnable frequency modulation.
    Pure sin/cos rotation — no torch.view_as_complex anywhere."""

    def __init__(self, dim, num_heads, base=10000):
        super().__init__()
        self.base = base
        self.num_heads = max(num_heads, 1)
        self.head_modulation = nn.Parameter(torch.ones(1, 1, 1, self.num_heads, 1))

    def forward(self, x):
        B, H, W, C = x.shape
        if C % 2 != 0:
            return x

        half = C // 2
        device, dtype = x.device, x.dtype

        k_h = max(half // 2, 1)
        k_w = half - k_h

        theta_h = 1.0 / (self.base ** (torch.arange(k_h, device=device, dtype=dtype) / max(k_h, 1)))
        theta_w = 1.0 / (self.base ** (torch.arange(k_w, device=device, dtype=dtype) / max(k_w, 1)))

        pos_h = torch.arange(H, device=device, dtype=dtype)
        pos_w = torch.arange(W, device=device, dtype=dtype)

        ang_h = pos_h.unsqueeze(-1) * theta_h.unsqueeze(0)
        ang_w = pos_w.unsqueeze(-1) * theta_w.unsqueeze(0)

        angles = torch.cat([
            ang_h[:, None, :].expand(H, W, k_h),
            ang_w[None, :, :].expand(H, W, k_w)
        ], dim=-1)

        cos_a = torch.cos(angles).unsqueeze(0)
        sin_a = torch.sin(angles).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = cos_a * x_even - sin_a * x_odd
        out_odd = sin_a * x_even + cos_a * x_odd

        x_rope = torch.stack([out_even, out_odd], dim=-1).reshape(B, H, W, C)

        hd = C // self.num_heads
        x_rope = x_rope.reshape(B, H, W, self.num_heads, hd)
        x_rope = x_rope * self.head_modulation
        return x_rope.reshape(B, H, W, C)


# ================================================================
#  S4: SwiGLU FFN
# ================================================================

class SwiGLU_FFN(nn.Module):
    """SwiGLU: (SiLU(xW_gate) * xW_up) W_down. Hidden = 8/3 dim."""

    def __init__(self, dim, drop=0.):
        super().__init__()
        hidden = ((int(dim * 8 / 3) + 7) // 8) * 8
        self.w_gate = nn.Linear(dim, hidden)
        self.w_up = nn.Linear(dim, hidden)
        self.w_down = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ================================================================
#  A6: RepDWConv — multi-scale context, free at inference
# ================================================================

class RepDWConv(nn.Module):
    """Training: parallel 3×3 + 5×5. Inference: merged into single 5×5."""

    def __init__(self, dim):
        super().__init__()
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.bn5 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn3(self.dw3(x)) + self.bn5(self.dw5(x))

    @torch.no_grad()
    def fuse(self):
        w3 = self.dw3.weight * (self.bn3.weight / (self.bn3.running_var + self.bn3.eps).sqrt()).view(-1, 1, 1, 1)
        b3 = self.bn3.bias - self.bn3.weight * self.bn3.running_mean / (self.bn3.running_var + self.bn3.eps).sqrt()
        w5 = self.dw5.weight * (self.bn5.weight / (self.bn5.running_var + self.bn5.eps).sqrt()).view(-1, 1, 1, 1)
        b5 = self.bn5.bias - self.bn5.weight * self.bn5.running_mean / (self.bn5.running_var + self.bn5.eps).sqrt()
        w3_padded = F.pad(w3, [1, 1, 1, 1])
        dim = w3.shape[0]
        fused = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=True)
        fused.weight.data = w3_padded + w5
        fused.bias.data = b3 + b5
        return fused


# ================================================================
#  FALA Linear Attention (S1 + S2 + S3)
# ================================================================

class FALALinearAttention(nn.Module):
    """Core attention: focused kernel (S1), WTConv LePE (S2), 2D RoPE-Mixed (S3)."""

    def __init__(self, dim, num_heads=4, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.lepe = WTConv2d(dim)                       # S2
        self.rope = RoPE2DMixed(dim, num_heads)          # S3

    def forward(self, x, H, W):
        B, N, C = x.shape
        num_heads = self.num_heads
        hd = self.head_dim

        qk = self.qk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x

        # S1: Focused kernel
        q = focused_kernel(q)
        k = focused_kernel(k)

        # S3: 2D RoPE-Mixed
        q_rope = self.rope(q.reshape(B, H, W, C)).reshape(B, N, num_heads, hd).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(B, H, W, C)).reshape(B, N, num_heads, hd).permute(0, 2, 1, 3)

        q = q.reshape(B, N, num_heads, hd).permute(0, 2, 1, 3)
        k = k.reshape(B, N, num_heads, hd).permute(0, 2, 1, 3)
        v = v.reshape(B, N, num_heads, hd).permute(0, 2, 1, 3)

        # Linear attention core
        z = 1.0 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        scale = N ** -0.5
        kv = (k_rope.transpose(-2, -1) * scale) @ (v * scale)
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(B, N, C)

        # S2: WTConv LePE
        v_2d = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.lepe(v_2d).permute(0, 2, 3, 1).reshape(B, N, C)

        return x


# ================================================================
#  FALABlock (S1-S3 via attention, S4 SwiGLU, A6 RepDWConv)
# ================================================================

class FALABlock(nn.Module):
    """Core block: S1-S4 + A6."""

    def __init__(self, dim, num_heads=None, mlp_ratio=4., qkv_bias=True,
                 drop=0., drop_path_rate=0., **kwargs):
        super().__init__()
        self.dim = dim
        num_heads = num_heads or max(1, dim // 32)
        self.num_heads = num_heads

        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = RepDWConv(dim)                        # A6
        self.act = nn.SiLU()
        self.attn = FALALinearAttention(dim, num_heads, qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU_FFN(dim, drop=drop)            # S4

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        x = x.permute(0, 2, 3, 1).reshape(B, N, C)

        # CPE
        x = x + self.cpe(
            x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).reshape(B, N, C)

        shortcut = x

        # Norm + split
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))

        # Main path: in_proj → RepDWConv → SiLU
        x = self.in_proj(x).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.act(self.dwc(x))
        x = x.permute(0, 2, 3, 1).reshape(B, N, C)

        # FALA attention (S1, S2, S3)
        x = self.attn(x, H, W)

        # Gate + output projection
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)

        # SwiGLU FFN (S4)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


# ================================================================
#  C3kFALA (inner shell + A8a)
# ================================================================

class C3kFALA(C3):
    """C3k with FALABlock inside + A8a inner gamma."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(FALABlock(c_) for _ in range(n)))
        self.gamma = nn.Parameter(0.01 * torch.ones(c2))

    def forward(self, x):
        main = self.m(self.cv1(x))
        skip = self.cv2(x)
        out = self.cv3(torch.cat((main, skip), 1))
        if x.shape == out.shape:
            out = out + self.gamma.view(1, -1, 1, 1) * x
        return out


# ================================================================
#  CSP_FALA (outer shell + A8b)
# ================================================================

class CSP_FALA(C2f):
    """CSP-FALA: replaces C3K2 in YOLO11.
    6 components: S1 S2 S3 S4 A6 A8."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3kFALA(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
        self.gamma = nn.Parameter(0.01 * torch.ones(c2))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        if x.shape == out.shape:
            out = out + self.gamma.view(1, -1, 1, 1) * x
        return out


# ================================================================
#  Tests
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CSP_FALA — 6 innovations")
    print("=" * 60)

    tests = [
        ("c3k=True  64ch  32x32",  1, 64,  32, 32, 64,  True),
        ("c3k=False 64ch  32x32",  1, 64,  32, 32, 64,  False),
        ("c3k=True  128ch 16x16",  1, 128, 16, 16, 128, True),
        ("c3k=True  256ch 8x8",    1, 256, 8,  8,  256, True),
        ("c1!=c2    64->128 16x16", 1, 64,  16, 16, 128, True),
        ("small ch  16ch  40x60",  1, 16,  40, 60, 16,  True),
        ("odd size  32ch  29x37",  1, 32,  29, 37, 32,  True),
    ]

    for name, B, c1, H, W, c2, c3k in tests:
        print(f"\n[Test] {name}")
        x = torch.rand(B, c1, H, W)
        model = CSP_FALA(c1, c2, n=1, c3k=c3k)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e3
        print(f"  In: {x.shape} -> Out: {out.shape}  Params: {params:.1f}K")
        assert out.shape == (B, c2, H, W), f"FAIL: {out.shape} != {(B, c2, H, W)}"
        print("  PASS")

    print(f"\n[Test] Gradient backward")
    x = torch.rand(1, 64, 16, 16, requires_grad=True)
    model = CSP_FALA(64, 64, n=1, c3k=True)
    loss = model(x).sum()
    loss.backward()
    assert x.grad is not None
    print(f"  Grad OK")
    print("  PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nInnovation inventory:")
    print("  S1: Focused kernel        — FALALinearAttention")
    print("  S2: WTConv LePE           — FALALinearAttention")
    print("  S3: 2D RoPE-Mixed         — FALALinearAttention")
    print("  S4: SwiGLU FFN            — FALABlock")
    print("  A6: RepDWConv 3x3+5x5     — FALABlock")
    print("  A8: Gamma residual        — C3kFALA (inner) + CSP_FALA (outer)")