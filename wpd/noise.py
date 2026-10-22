"""
ψ-PD structured noise construction.

The source latent x and a Gaussian noise sample ϵ are decomposed with the Dual-Tree
Complex Wavelet Packet Transform (DT-CWPT). For every complex high-frequency packet,
the source phase is injected where the packet's mid frequency lies below the (possibly
spatially varying) cutoff map F, while the magnitude is always taken from the noise.
The real-valued low-frequency packet L is either randomized (Low-Frequency
Randomization, LFR) or, without LFR, fused via global Fourier phase injection.
The inverse transform yields the structured noise ϵ̂ that replaces ϵ in the diffusion
forward process (training) and as the initial latent (inference).
"""
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse


# ---------------------------------------------------------------------------
# Complex-coefficient helpers
# ---------------------------------------------------------------------------
def _resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if x.shape[-2:] == (h, w):
        return x
    return F.interpolate(x.float(), size=(h, w), mode="bilinear", align_corners=False)


def _fuse_phase(c_src: torch.Tensor, c_noise: torch.Tensor, mask, eps: float = 1e-8) -> torch.Tensor:
    """Combine source phase (mask=1) / noise phase (mask=0) with the noise magnitude.

    c_src, c_noise: (N, C, H, W, 2) complex coefficients stored as (re, im).
    """
    phi_s = torch.atan2(c_src[..., 1], c_src[..., 0])
    phi_n = torch.atan2(c_noise[..., 1], c_noise[..., 0])
    phi = mask * phi_s + (1.0 - mask) * phi_n
    mag = torch.sqrt(c_noise[..., 0] ** 2 + c_noise[..., 1] ** 2 + eps)
    return torch.stack([mag * torch.cos(phi), mag * torch.sin(phi)], dim=-1)


def _cplx_to_chan(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[..., 0], x[..., 1]], dim=1)


def _chan_to_cplx(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1] // 2
    return torch.stack([x[:, :c], x[:, c:]], dim=-1)


def _fourier_phase_ll(ll_src: torch.Tensor, ll_noise: torch.Tensor, pad_factor: float = 1.5) -> torch.Tensor:
    """Low-frequency packet without LFR: |FFT(noise)| * exp(j * angle(FFT(src))), reflect-padded."""
    H, W = ll_src.shape[-2:]
    pad_h = int(H * (pad_factor - 1)) // 2 * 2
    pad_w = int(W * (pad_factor - 1)) // 2 * 2
    pad = (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2)
    if pad_h or pad_w:
        ll_src = F.pad(ll_src, pad, mode="reflect")
        ll_noise = F.pad(ll_noise, pad, mode="reflect")

    fft_src = torch.fft.fft2(ll_src, dim=(-2, -1))
    fft_noise = torch.fft.fft2(ll_noise, dim=(-2, -1))
    fft_mix = torch.abs(fft_noise) * torch.exp(1j * torch.angle(fft_src))
    ll_mix = torch.fft.ifft2(fft_mix, dim=(-2, -1)).real
    return ll_mix[..., pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W]


# ---------------------------------------------------------------------------
# DT-CWPT phase injection
# ---------------------------------------------------------------------------
class _PacketSplitter(nn.Module):
    """Split a complex subband into its low / high halves with one more DT-CWT level."""

    def __init__(self, biort: str, qshift: str):
        super().__init__()
        self.fwd = DTCWTForward(J=1, biort=biort, qshift=qshift, mode="symmetric", o_dim=2, ri_dim=-1)
        self.inv = DTCWTInverse(biort=biort, qshift=qshift, mode="symmetric", o_dim=2, ri_dim=-1)

    def forward(self, band: torch.Tensor):
        h, w = band.shape[-3], band.shape[-2]
        yl, yh = self.fwd(_cplx_to_chan(band))
        low = _chan_to_cplx(self.inv((yl, [torch.zeros_like(yh[0])])))[..., :h, :w, :]
        high = _chan_to_cplx(self.inv((torch.zeros_like(yl), yh)))[..., :h, :w, :]
        return low, high


class DTCWPTPhaseInjection(nn.Module):
    def __init__(self, J: int, biort: str = "near_sym_b", qshift: str = "qshift_b", max_packet_level: int = 4):
        super().__init__()
        self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift, mode="symmetric", o_dim=2, ri_dim=-1)
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift, mode="symmetric", o_dim=2, ri_dim=-1)
        self.splitter = _PacketSplitter(biort, qshift)
        self.max_packet_level = max_packet_level

    def forward(self, x: torch.Tensor, noise: torch.Tensor, cutoff: torch.Tensor,
                lfr: bool = True, pad_factor: float = 1.5) -> torch.Tensor:
        """
        x, noise: (N, C, H, W) source latent and Gaussian noise.
        cutoff:   (N, 1, H, W) normalized cutoff map F in [0, 1].
        """
        ll_x, yh_x = self.xfm(x)
        ll_z, yh_z = self.xfm(noise)

        # Low-frequency packet: randomized (LFR) or Fourier phase injection.
        ll_mix = ll_z if lfr else _fourier_phase_ll(ll_x, ll_z, pad_factor)

        yh_mix = []
        for level, (cx, cz) in enumerate(zip(yh_x, yh_z), start=1):
            f_high = 1.0 / 2 ** (level - 1)
            f_low = 1.0 / 2 ** level
            N, C, O, H, W, _ = cx.shape
            f_map = _resize(cutoff, H, W)
            fused = self._inject(
                cx.contiguous().view(N, C * O, H, W, 2),
                cz.contiguous().view(N, C * O, H, W, 2),
                f_low, f_high, f_map, 0,
            )
            yh_mix.append(fused.view(N, C, O, H, W, 2))

        return self.ifm((ll_mix, yh_mix))

    def _inject(self, c_x, c_z, f_start, f_end, f_map, level):
        # Entire packet below the cutoff everywhere -> source phase; above everywhere -> noise phase.
        if f_end <= f_map.min().item() + 1e-9:
            return _fuse_phase(c_x, c_z, 1.0)
        if f_start >= f_map.max().item() - 1e-9:
            return _fuse_phase(c_x, c_z, 0.0)

        mid = (f_start + f_end) / 2
        if level >= self.max_packet_level:
            return _fuse_phase(c_x, c_z, (mid <= f_map).float())

        # Mixed packet: split further into wavelet packets and recurse.
        lo_x, hi_x = self.splitter(c_x)
        lo_z, hi_z = self.splitter(c_z)
        sub_map = _resize(f_map, lo_x.shape[-3], lo_x.shape[-2])
        return (self._inject(lo_x, lo_z, f_start, mid, sub_map, level + 1)
                + self._inject(hi_x, hi_z, mid, f_end, sub_map, level + 1))


_module_cache: dict = {}


def _get_module(J: int, biort: str, qshift: str, device) -> DTCWPTPhaseInjection:
    key = (J, biort, qshift, str(device))
    if key not in _module_cache:
        _module_cache[key] = DTCWPTPhaseInjection(J, biort, qshift).to(device)
    return _module_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def radius_to_cutoff(radius: Union[float, torch.Tensor], height: int, width: int) -> Union[float, torch.Tensor]:
    """Convert a cutoff radius (in latent pixels) to a Nyquist-normalized cutoff."""
    nyquist = min(height, width) / 2.0
    if isinstance(radius, torch.Tensor):
        return radius.float() / nyquist
    return max(float(radius), 1.0) / nyquist


def auto_J(cutoff: float) -> int:
    """Smallest decomposition depth whose finest packet split resolves the cutoff."""
    return max(1, math.ceil(-math.log2(max(cutoff, 1e-2))))


@torch.no_grad()
def wpd_noise(
    latents: torch.Tensor,
    cutoff: Union[float, torch.Tensor] = 0.5,
    J: Optional[int] = 4,
    lfr: bool = True,
    noise: Optional[torch.Tensor] = None,
    biort: str = "near_sym_b",
    qshift: str = "qshift_b",
) -> torch.Tensor:
    """Generate ψ-PD structured noise.

    Args:
        latents: (N, C, H, W) source latents. For video latents, pass frames as the batch dim.
        cutoff:  Normalized cutoff F in [0, 1] (1 = Nyquist). A float for a global cutoff, or an
                 (N, 1, H, W) map for spatially adaptive / instance-level translation. Source phase
                 is preserved below the cutoff; higher cutoff = more structure preserved.
        J:       DT-CWPT decomposition depth. None = derived from the minimum cutoff.
        lfr:     Low-Frequency Randomization: replace the low-frequency packet with noise.
        noise:   Optional Gaussian noise sample (same shape as latents).
    """
    if latents.ndim != 4:
        raise ValueError(f"Expected (N, C, H, W) latents, got {tuple(latents.shape)}")
    dtype, device = latents.dtype, latents.device
    x = latents.float()
    N, _, H, W = x.shape

    if isinstance(cutoff, torch.Tensor):
        cutoff_map = cutoff.to(device).float()
    else:
        cutoff_map = torch.full((N, 1, H, W), float(cutoff), device=device)
    cutoff_map = cutoff_map.clamp(0.0, 1.0)

    z = torch.randn_like(x) if noise is None else noise.to(device).float()
    if J is None:
        J = auto_J(cutoff_map.min().item())

    x_hat = _get_module(max(1, J), biort, qshift, device)(x, z, cutoff_map, lfr=lfr)
    # Symmetric wavelet padding can leave an extra row or column for odd latent sizes.
    x_hat = x_hat[..., :H, :W]
    # Guard against rare numerical outliers from the recursive packet reconstruction.
    x_hat = torch.where(x_hat.abs() > 5, z, x_hat)
    return x_hat.to(dtype)


def wpd_video_noise(latents: torch.Tensor, **kwargs) -> torch.Tensor:
    """Per-meta-frame ψ-PD noise for video latents shaped (1, C, T, H, W)."""
    frames = latents[0].transpose(0, 1)  # (T, C, H, W)
    return wpd_noise(frames, **kwargs).transpose(0, 1).unsqueeze(0).contiguous()
