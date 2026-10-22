"""FLUX.1-dev (image) and Wan2.2-I2V-A14B (video) ψ-PD inference helpers."""
import gc
import os
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig as FluxModelConfig
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig as WanModelConfig

from .noise import wpd_noise, wpd_video_noise


FLUX_NEGATIVE_PROMPT = (
    "ugly, low quality, CG, render, unreal, game, cartoon, blur, low res, "
    "dashboard, steering wheel, windshield frame, car interior, lens artifacts"
)
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，卡通，渲染，游戏，CG，render, simulation, game, cartoon, 3D"
)
VAE_SCALE = 8  # spatial downsampling factor of both the FLUX and Wan VAEs
SPATIAL_DIVISOR = 16  # required by both inference pipelines
DEFAULT_LONG_SIDE = 1280
MAX_VIDEO_FRAMES = 49


def align_resolution(height: int, width: int) -> tuple[int, int]:
    """Match the spatial rounding performed by the FLUX and Wan pipelines."""
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be positive.")
    return tuple((size + SPATIAL_DIVISOR - 1) // SPATIAL_DIVISOR * SPATIAL_DIVISOR
                 for size in (height, width))


def resolve_resolution(source_height: int, source_width: int,
                       height: Optional[int] = None, width: Optional[int] = None) -> tuple[int, int]:
    """Use a 1280-pixel long side by default, preserving aspect ratio as closely as possible."""
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Source image height and width must be positive.")
    if height is not None or width is not None:
        return align_resolution(
            source_height if height is None else height,
            source_width if width is None else width,
        )
    short = min(source_height, source_width) * DEFAULT_LONG_SIDE / max(source_height, source_width)
    short = max(SPATIAL_DIVISOR, int(short / SPATIAL_DIVISOR + 0.5) * SPATIAL_DIVISOR)
    return (DEFAULT_LONG_SIDE, short) if source_height >= source_width else (short, DEFAULT_LONG_SIDE)


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def _check(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path} (see README for download instructions)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_flux(device: str, lora: str, vram_management: bool = False) -> FluxImagePipeline:
    _check(lora)
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )
    pipe.load_lora(pipe.dit, lora, alpha=1.0)
    if vram_management:
        pipe.enable_vram_management()
    return pipe


def load_wan(device: str, high_lora: str, low_lora: str,
             lightning_high: str, lightning_low: str) -> WanVideoPipeline:
    for path in (high_lora, low_lora, lightning_high, lightning_low):
        _check(path)
    model_id = "Wan-AI/Wan2.2-I2V-A14B"
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            WanModelConfig(model_id=model_id, origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            WanModelConfig(model_id=model_id, origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            WanModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            WanModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    # ψ-PD LoRAs for the high-/low-noise experts, then the 4-step Wan2.2-Lightning LoRAs.
    pipe.load_lora(pipe.dit, high_lora, alpha=1.0)
    pipe.load_lora(pipe.dit2, low_lora, alpha=1.0)
    pipe.load_lora(pipe.dit, lightning_high, alpha=8.0 / 64)
    pipe.load_lora(pipe.dit2, lightning_low, alpha=8.0 / 64)
    pipe.enable_vram_management()
    return pipe


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_image(path: str, height: Optional[int] = None, width: Optional[int] = None) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    height, width = resolve_resolution(image.height, image.width, height, width)
    return image.resize((width, height), Image.LANCZOS) if image.size != (width, height) else image


def load_video(path: str, height: Optional[int] = None, width: Optional[int] = None):
    """Return up to 49 PIL frames and their fps; reject longer videos."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        target_size = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frames) == MAX_VIDEO_FRAMES:
                raise ValueError(f"Input video exceeds {MAX_VIDEO_FRAMES} frames: {path}")
            if target_size is None:
                source_height, source_width = frame.shape[:2]
                target_height, target_width = resolve_resolution(source_height, source_width, height, width)
                target_size = (target_width, target_height)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(image.resize(target_size, Image.LANCZOS) if image.size != target_size else image)
    finally:
        cap.release()
    return frames, fps


def cutoff_map_from_mask(mask_path: str, height: int, width: int, strength: float = 0.5,
                         device: str = "cuda") -> torch.Tensor:
    """Build a spatial cutoff map F for instance-level translation.

    White mask pixels are translated (cutoff = 1 - strength), black pixels keep the source
    phase up to Nyquist (cutoff = 1) and thus stay untranslated. Returns (1, 1, H/8, W/8).
    """
    h_lat, w_lat = height // VAE_SCALE, width // VAE_SCALE
    mask = Image.open(mask_path).convert("L").resize((width, height), Image.BILINEAR)
    mask = np.array(mask.resize((w_lat, h_lat), Image.BILINEAR), dtype=np.float32) / 255.0
    cutoff = 1.0 - mask * strength
    return torch.from_numpy(cutoff)[None, None].to(device)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
@torch.no_grad()
def translate_image(
    pipe: FluxImagePipeline,
    image: Image.Image,
    prompt: str,
    cutoff: Union[float, torch.Tensor] = 0.5,
    J: Optional[int] = 4,
    lfr: bool = True,
    negative_prompt: str = FLUX_NEGATIVE_PROMPT,
    cfg_scale: float = 2.0,
    num_inference_steps: int = 50,
) -> Image.Image:
    device = pipe.device
    image_tensor = pipe.preprocess_image(image).to(device=device, dtype=pipe.torch_dtype)
    latents = pipe.vae_encoder(image_tensor, tiled=False).to(device)
    noise = wpd_noise(latents, cutoff=cutoff, J=J, lfr=lfr).contiguous()
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=image.height, width=image.width,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        noise=noise,
    )


@torch.no_grad()
def translate_video(
    pipe: WanVideoPipeline,
    frames: List[Image.Image],
    first_frame: Image.Image,
    prompt: str,
    cutoff: float = 0.5,
    J: Optional[int] = 4,
    lfr: bool = True,
    negative_prompt: str = WAN_NEGATIVE_PROMPT,
    num_inference_steps: int = 4,
) -> List[Image.Image]:
    """Translate at most 49 frames with Wan I2V, conditioned on the translated first frame."""
    device = pipe.device
    height, width = first_frame.height, first_frame.width
    total = len(frames)
    if not 1 <= total <= MAX_VIDEO_FRAMES:
        raise ValueError(f"Expected 1–{MAX_VIDEO_FRAMES} frames, got {total}.")
    chunk = frames + [frames[-1]] * (MAX_VIDEO_FRAMES - total)

    pipe.load_models_to_device(["vae"])
    pixels = pipe.preprocess_video(chunk).to(device=device, dtype=torch.bfloat16)
    latents = pipe.vae.encode(pixels, device=device, tiled=True)
    noise = wpd_video_noise(latents.float(), cutoff=cutoff, J=J, lfr=lfr)

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        tiled=True,
        input_image=first_frame,
        input_noise=noise.to(dtype=pipe.torch_dtype, device=device),
        height=height, width=width,
        num_frames=MAX_VIDEO_FRAMES,
        switch_DiT_boundary=0.9,
        cfg_scale=1,
        num_inference_steps=num_inference_steps,
    )
    return video[:total]


