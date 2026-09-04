import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange, repeat
from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    SAGE_ATTN_AVAILABLE = False

try:
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    SPARGE_ATTN_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    SPARGE_ATTN_AVAILABLE = False


_ATTENTION_BACKENDS = ("flash_attn_3", "flash_attn_2", "sageattention", "sdpa")
_ATTENTION_BACKEND_ENV = "FDANYONE_ATTENTION_BACKEND"
_ATTENTION_BACKEND = None


def _select_attention_backend() -> str:
    """Local patch: allow FDANYONE_ATTENTION_BACKEND to override the auto policy.

    The auto policy ranks by speed, but on a 32 GB card this pipeline is bound
    by memory, not by time. Measured on an RTX 5090 (bf16, 24 heads, head_dim
    128) sageattn peaks at exactly 2x the memory of torch SDPA -- it keeps
    INT8 copies of q and k plus a smoothed k alongside the originals, whereas
    SDPA already dispatches to the flash kernel and is O(N). Numbers for the
    multiview-attention shape this model uses:

        v=5 (RCP off)   SDPA +1.57 GiB  36 ms   sage +3.13 GiB  22 ms
        v=6 (RCP on)    SDPA +1.89 GiB  50 ms   sage +3.75 GiB  31 ms

    So on any machine where sageattention happens to be importable -- notably
    a shared ComfyUI environment -- the auto policy silently costs ~1.9 GiB at
    the exact moment RCP needs it. Set the variable to "sdpa" to opt out.
    """

    override = os.environ.get(_ATTENTION_BACKEND_ENV, "").strip().lower()
    if override:
        if override not in _ATTENTION_BACKENDS:
            raise ValueError(
                f"{_ATTENTION_BACKEND_ENV} must be one of {', '.join(_ATTENTION_BACKENDS)}, got {override!r}."
            )
        available = {
            "flash_attn_3": FLASH_ATTN_3_AVAILABLE,
            "flash_attn_2": FLASH_ATTN_2_AVAILABLE,
            "sageattention": SAGE_ATTN_AVAILABLE,
            "sdpa": True,
        }
        if not available[override]:
            raise ValueError(f"{_ATTENTION_BACKEND_ENV}={override!r} but that backend is not importable.")
        return override

    if FLASH_ATTN_3_AVAILABLE:
        return "flash_attn_3"
    if FLASH_ATTN_2_AVAILABLE:
        return "flash_attn_2"
    if SAGE_ATTN_AVAILABLE:
        return "sageattention"
    return "sdpa"


def get_attention_backend() -> str:
    """Return the implementation selected by the vendored auto policy."""

    global _ATTENTION_BACKEND
    if _ATTENTION_BACKEND is None:
        _ATTENTION_BACKEND = _select_attention_backend()
    return _ATTENTION_BACKEND


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    # Local patch: dispatch on the resolved backend name rather than on the
    # availability flags, so FDANYONE_ATTENTION_BACKEND actually takes effect.
    # With the variable unset this reproduces the original branch order.
    backend = get_attention_backend()
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif backend == "flash_attn_3":
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif backend == "flash_attn_2":
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    # TODO: test spas_sage2_attn_meansim_topk_cuda
    # elif SPARGE_ATTN_AVAILABLE:
    #     q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    #     k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    #     v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    #     x = spas_sage2_attn_meansim_topk_cuda(q, k, v, simthreshd1=-0.1, topk=0.5, pvthreshd=15, is_causal=False)
    #     x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif backend == "sageattention":
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _chunked_tokenwise(module, x, max_tokens=16384):
    """Local patch: run a per-token module (the FFN) in sequence slices.

    The FFN intermediate (tokens x ffn_dim) was a single 3.64 GiB allocation
    at full sequence length. Every layer inside is pointwise per token, so
    slicing the sequence dimension is bit-identical and bounds the temporary
    to max_tokens rows."""
    s = x.shape[1]
    if s <= max_tokens:
        return module(x)
    return torch.cat([module(x[:, i:i + max_tokens]) for i in range(0, s, max_tokens)], dim=1)


def rope_apply(x, freqs, num_heads):
    # Local patch: the original upcast the WHOLE tensor to float64 (4x bf16)
    # and built an equally large complex product -- a 3.12 GiB single
    # allocation that OOMed a 32 GB card. RoPE is elementwise along the
    # sequence dim (freqs indexes it), so apply it in sequence chunks with
    # identical float64 math and write results into a bf16-sized output.
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    b, s, n, d = x.shape
    out = torch.empty_like(x)
    # ~64M float64 elements per slice keeps the temporary under ~1 GiB
    chunk = max(1, (1 << 26) // max(1, b * n * d))
    for i in range(0, s, chunk):
        piece = x[:, i:i + chunk].to(torch.float64)
        piece = torch.view_as_complex(piece.reshape(b, piece.shape[1], n, -1, 2))
        out[:, i:i + chunk] = torch.view_as_real(piece * freqs[i:i + chunk]).flatten(3).to(x.dtype)
    return out.flatten(2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        # Local patch: norm is pointwise per token, but x.float() upcast the
        # whole (b, s, d) tensor at once -- a 1.88 GiB single allocation at
        # 24-view sequence lengths. Slice the sequence dim; bit-identical.
        # NB: the trailing multiply PROMOTES back to the weight's float32, so
        # slices are concatenated rather than written into a bf16 buffer.
        if x.dim() == 3 and x.shape[1] > 16384:
            return torch.cat(
                [self.norm(x[:, i:i + 16384].float()).to(dtype) * self.weight
                 for i in range(0, x.shape[1], 16384)], dim=1)
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class CrossAttentionSrcCam(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, x_src, freqs, freqs_src):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x_src))
        v = self.v(x_src)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs_src, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.disable_video_attn = False
        self.use_4d_attn = False
        self.use_mvs_attn = False
        self.use_src_self_attn = False
        self.use_src_cross_attn = False
        self.use_cam_encoder = False

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, x_src, context, t_mod, freqs, freqs_mvs, freqs_src, cam_emb, shape):
        v, f, h, w = shape

        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        # video self-attention
        if not self.disable_video_attn:
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
            if self.use_4d_attn:
                input_x = rearrange(input_x, "v fhw c -> (v fhw) c").unsqueeze(0)
                freqs = repeat(freqs, "fhw 1 c -> (v fhw) 1 c", v=v)
            input_x = self.self_attn(input_x, freqs)
            if self.use_4d_attn:
                input_x = rearrange(input_x.squeeze(0), "(v fhw) c -> v fhw c", v=v)
            x = self.gate(x, gate_msa, input_x)

        # source-view self-attention
        if self.use_src_self_attn:
            x_cat = torch.cat([x, x_src], dim=1)
            shift_src, scale_src, gate_src = (
                self.modulation_src.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod[:, :3, :]).chunk(3, dim=1)
            input_x_cat = modulate(self.norm1_src(x_cat), shift_src, scale_src)

            freqs_cat = torch.cat([freqs, freqs_src], dim=0)
            input_x_cat = self.self_attn_src(input_x_cat, freqs_cat)
            x_cat = self.gate(x_cat, gate_src, input_x_cat)
            len_src = x_src.shape[1]
            x, x_src = x_cat[:, :-len_src, ...], x_cat[:, -len_src:, ...]

        # source-view cross-attention
        if self.use_src_cross_attn:
            shift_src, scale_src, gate_src = (
                self.modulation_src.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod[:, :3, :]).chunk(3, dim=1)
            input_x = modulate(self.norm1_src(x), shift_src, scale_src)
            input_x_src = self.norm1_src(x_src)

            input_x = self.cross_attn_src(input_x, input_x_src, freqs, freqs_src)
            x = self.gate(x, gate_src, input_x)

        # multiview self-attention
        if self.use_mvs_attn:
            shift_mvs, scale_mvs, gate_mvs = (
                self.modulation_mvs.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod[:, :3, :]).chunk(3, dim=1)
            input_x = modulate(self.norm1_mvs(x), shift_mvs, scale_mvs)

            # add camera embedding before multiview attention
            if self.use_cam_encoder and cam_emb is not None:
                cam_proj = self.cam_encoder(cam_emb)  # (v, 1, dim)
                cam_proj = cam_proj.unsqueeze(2).unsqueeze(3).expand(-1, f, h, w, -1)  # (v, f, h, w, dim)
                cam_proj = rearrange(cam_proj, "v f h w d -> v (f h w) d")
                input_x = input_x + cam_proj

            input_x = rearrange(input_x, "v (f h w) c -> f (v h w) c", v=v, f=f, h=h, w=w)
            input_x = self.self_attn_mvs(input_x, freqs_mvs)
            input_x = rearrange(input_x, "f (v h w) c -> v (f h w) c", v=v, f=f, h=h, w=w)

            # projector wraps multiview attention output
            if self.use_cam_encoder:
                input_x = self.projector(input_x)

            x = self.gate(x, gate_mvs, input_x)

        # prompt cross-attention
        context = repeat(context, "1 l c -> v l c", v=x.shape[0])
        x = x + self.cross_attn(self.norm3(x), context)

        # feed-forward network
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, _chunked_tokenwise(self.ffn, input_x))

        if self.use_src_self_attn:
            # for src self-attention, the x_src is updated as well
            x_src = x_src + self.cross_attn(self.norm3(x_src), context)

            input_x_src = modulate(self.norm2(x_src), shift_mlp, scale_mlp)
            x_src = self.gate(x_src, gate_mlp, _chunked_tokenwise(self.ffn, input_x_src))

        return x, x_src


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            if t_mod.shape[0] != 1:
                t_mod = t_mod[:, None, :] # [b, d] -> [b, 1, d]
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


def pad_for_3d_conv(x, kernel_size):
    """Pad to be divisible by kernel_size. From FramePack."""
    _, _, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')


class ViewPackEmbedding(nn.Module):
    """Multi-resolution spatial patch embedding for clean source views.
    Spatial-only downsampling: temporal dim preserved.
    Ref: HunyuanVideoPatchEmbedForCleanLatents in FramePack.

    Supported configurations (all exactly fill 4 quadrants):
      - 4×2x views  (each 2x view → 1 quadrant)
      - 3×2x + 4×4x  (3 quadrants from 2x + 1 quadrant from 4×4x tile)
    """

    def __init__(self, in_dim, dim, patch_size):
        super().__init__()
        pt, ph, pw = patch_size  # (1, 2, 2) for Wan
        # 2x: spatial 2x downsample relative to 1x -> kernel (1, 4, 4)
        self.proj_2x = nn.Conv3d(in_dim, dim, kernel_size=(pt, ph*2, pw*2), stride=(pt, ph*2, pw*2))
        # 4x: spatial 4x downsample relative to 1x -> kernel (1, 8, 8)
        self.proj_4x = nn.Conv3d(in_dim, dim, kernel_size=(pt, ph*4, pw*4), stride=(pt, ph*4, pw*4))

    @torch.no_grad()
    def initialize_from_patch_embedding(self, patch_embedding: nn.Conv3d):
        """FramePack-style init: tile spatial dims and scale by 1/area_ratio."""
        weight = patch_embedding.weight.detach().clone()  # (dim, in_dim, 1, 2, 2)
        bias = patch_embedding.bias.detach().clone()
        sd = {
            'proj_2x.weight': repeat(weight, 'b c t h w -> b c t (h 2) (w 2)') / 4.0,
            'proj_2x.bias': bias.clone(),
            'proj_4x.weight': repeat(weight, 'b c t h w -> b c t (h 4) (w 4)') / 16.0,
            'proj_4x.bias': bias.clone(),
        }
        self.load_state_dict(sd)


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                x_src: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                skeletons: Optional[torch.Tensor] = None,
                cam_emb: Optional[torch.Tensor] = None,
                drop_viewpack_tokens: bool = False,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # build packed views from 1x/2x/4x source views
        v_src = x_src.shape[0]
        if v_src == 1:
            # 1×1x: no 2x/4x sources
            x_src_2x, x_src_4x = None, None
        elif v_src == 5:
            # 1×1x + 4×2x
            x_src, x_src_2x, x_src_4x = x_src[:1], x_src[1:], None
        elif v_src == 8:
            # 1×1x + 3×2x + 4×4x
            x_src, x_src_2x, x_src_4x = x_src[:1], x_src[1:4], x_src[4:]
        else:
            raise ValueError(f"Unsupported number of source views: {v_src}")

        # 1x source: always packed as one extra view
        x_src, (f_src, h_src, w_src) = self.patchify(x_src)
        if not self.use_src_attn:
            # use viewpack for 1x source views
            x = torch.cat([x, x_src], dim=0)
            x_src = None
            freqs_src = None
            v_pack = 1
        else:
            # use src self/cross-attention for 1x source views
            x_src = repeat(x_src, "1 fhw_src c -> v fhw_src c", v=x.shape[0])
            freqs_src = torch.cat([
                self.freqs[0][self.freqs_src_shift:f_src + self.freqs_src_shift].view(f_src, 1, 1, -1).expand(f_src, h_src, w_src, -1),
                self.freqs[1][:h_src].view(1, h_src, 1, -1).expand(f_src, h_src, w_src, -1),
                self.freqs[2][:w_src].view(1, 1, w_src, -1).expand(f_src, h_src, w_src, -1)
            ], dim=-1).reshape(f_src * h_src * w_src, 1, -1).to(x.device)
            v_pack = 0

        # 2x/4x source views: packed after 1x source views
        if self.use_viewpack:
            if x_src_2x is not None:
                x_src_2x = pad_for_3d_conv(x_src_2x, self.viewpack_embedding.proj_2x.kernel_size)
                x_src_2x = self.viewpack_embedding.proj_2x(x_src_2x)  # (v_2x, dim, f, h//2, w//2)

                if x_src_4x is not None:
                    # tile 4x source views into 2x source views
                    x_src_4x = pad_for_3d_conv(x_src_4x, self.viewpack_embedding.proj_4x.kernel_size)
                    x_src_4x = self.viewpack_embedding.proj_4x(x_src_4x)  # (4, dim, f, h//4, w//4)
                    x_src_4x = rearrange(x_src_4x, '(g1 g2) c f h w -> 1 c f (g1 h) (g2 w)', g1=2, g2=2)  # (1, dim, f, h//2, w//2)
                    f_2x, h_2x, w_2x = x_src_2x.shape[2:] # crop padding surplus to match 2x source views (f, h, w)
                    x_src_4x = x_src_4x[:, :, :f_2x, :h_2x, :w_2x]
                    x_src_2x = torch.cat([x_src_2x, x_src_4x], dim=0)  # (4, dim, f, h//2, w//2)

                # tile 2x source views into 1x source views
                x_pack = rearrange(x_src_2x, '(g1 g2) c f h w -> 1 c f (g1 h) (g2 w)', g1=2, g2=2)
                x_pack = x_pack[:, :, :f, :h, :w] # crop padding surplus to match 1x source views (f, h, w)
                x_pack = rearrange(x_pack, '1 c f h w -> 1 (f h w) c')
                x_pack = x_pack.to(dtype=x.dtype)
                if drop_viewpack_tokens:
                    # Keep viewpack parameters in the autograd graph across distributed ranks.
                    zero_dependency = x_pack.float().mean().to(dtype=x.dtype) * 0.0
                    x = x + zero_dependency
                else:
                    x = torch.cat([x, x_pack], dim=0)
                    v_pack += 1

        timestep = torch.cat([timestep, torch.zeros(v_pack, device=timestep.device, dtype=timestep.dtype)])
        if skeletons is not None:
            skeletons = torch.cat([skeletons, -torch.ones_like(skeletons[:1]).expand(v_pack, -1, -1, -1, -1)], dim=0)

        # Expand cam_emb for viewpack views (zero vectors for packed source views)
        if cam_emb is not None:
            cam_emb = torch.cat([cam_emb, torch.zeros(v_pack, cam_emb.shape[-1],
                device=cam_emb.device, dtype=cam_emb.dtype)], dim=0)
            cam_emb = cam_emb.unsqueeze(1)  # (v, 1, 12)

        # Compute time embeddings (after concat since timestep may have been extended)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        if self.use_pose_encoder:
            # Local patch: encode per view instead of one all-views batch. The
            # batched Conv3d activations peaked at 4.37 GiB, which OOMed a
            # 32 GB card; per-view chunks cut the peak ~7x for negligible cost
            # (the encoder is a small conv stack inside the denoise loop).
            skeleton_latents = torch.cat(
                [self.pose_encoder(skeletons[i:i + 1]) for i in range(skeletons.shape[0])],
                dim=0)
            skeleton_tokens = rearrange(skeleton_latents, "v c f h w -> v (f h w) c")
            x = x + skeleton_tokens

        v = x.shape[0]
        freqs_mvs = torch.cat([
            self.freqs[0][:v].view(v, 1, 1, -1).expand(v, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(v, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(v, h, w, -1)
        ], dim=-1).reshape(v * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_src = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, x_src, context, t_mod, freqs, freqs_mvs, freqs_src, cam_emb, (v, f, h, w),
                            use_reentrant=False,
                        )
                else:
                    x, x_src = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, x_src, context, t_mod, freqs, freqs_mvs, freqs_src, cam_emb, (v, f, h, w),
                        use_reentrant=False,
                    )
            else:
                x, x_src = block(x, x_src, context, t_mod, freqs, freqs_mvs, freqs_src, cam_emb, (v, f, h, w))

        # Strip added views (viewpack)
        if v_pack > 0:
            x = x[:-v_pack, ...]
            t = t[:-v_pack, ...]

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()

    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            # 1.3B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            # 14B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        elif hash_state_dict_keys(state_dict) == "70ddad9d3a133785da5ea371aae09504":
            # 1.3B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "26bde73488a92e64cc20b0a7485b9e5b":
            # 14B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "ac6a5aa74f4a0aab6f64eb9a72f19901":
            # 1.3B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            # 14B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "1f5ab7703c6fc803fdded85ff040c316":
            # Wan-AI/Wan2.2-TI2V-5B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
            }
        elif hash_state_dict_keys(state_dict) == "5b013604280dd715f8457c6ed6d6a626":
            # Wan-AI/Wan2.2-I2V-A14B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "2267d489f0ceb9f21836532952852ee5":
            # Wan2.2-Fun-A14B-Control
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 52,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "47dbeab5e560db3180adf51dc0232fb1":
            # Wan2.2-Fun-A14B-Control-Camera
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
                "require_clip_embedding": False,
            }
        else:
            config = {}
        return state_dict, config
