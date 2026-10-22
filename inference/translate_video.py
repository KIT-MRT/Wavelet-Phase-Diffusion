"""
ψ-PD two-stage video sim-to-real translation.

Stage 1 translates the first frame with FLUX.1-dev into a photorealistic reference.
Stage 2 translates the whole video with Wan2.2-I2V-A14B conditioned on that reference.

Translate a single video:
    python inference/translate_video.py --input sim.mp4 --output real.mp4
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffsynth import save_video
from wpd.pipelines import (FLUX_NEGATIVE_PROMPT, WAN_NEGATIVE_PROMPT, cutoff_map_from_mask, flush,
                           load_flux, load_video, load_wan, translate_image,
                           translate_video)

DEFAULT_PROMPT = (
    "A photorealistic driving scene filmed from a moving vehicle. Natural lighting, urban buildings, "
    "trees, cars on the street. High resolution, realistic textures."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output video path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--mask", default=None,
                        help="Instance mask for the first frame (white = translate).")
    parser.add_argument("--mask_strength", type=float, default=0.5)
    # Stage 1: FLUX
    parser.add_argument("--flux_lora", default="checkpoints/flux.safetensors")
    parser.add_argument("--flux_cutoff", type=float, default=0.5, help="Fraction of the latent Nyquist frequency.")
    parser.add_argument("--flux_J", type=int, default=4, help="DT-CWPT depth (0 = auto).")
    parser.add_argument("--flux_no_lfr", action="store_true")
    parser.add_argument("--flux_negative_prompt", default=FLUX_NEGATIVE_PROMPT)
    # Stage 2: Wan
    parser.add_argument("--wan_high_lora", default="checkpoints/wan_high.safetensors")
    parser.add_argument("--wan_low_lora", default="checkpoints/wan_low.safetensors")
    parser.add_argument("--lightning_high", default="checkpoints/high_noise_model_converted.safetensors")
    parser.add_argument("--lightning_low", default="checkpoints/low_noise_model_converted.safetensors")
    parser.add_argument("--wan_cutoff", type=float, default=0.5, help="Fraction of the latent Nyquist frequency.")
    parser.add_argument("--wan_J", type=int, default=4, help="DT-CWPT depth (0 = auto).")
    parser.add_argument("--wan_no_lfr", action="store_true")
    parser.add_argument("--wan_negative_prompt", default=WAN_NEGATIVE_PROMPT)
    # General
    parser.add_argument("--height", type=int, default=None, help="Output height (default: scale source to long side 1280).")
    parser.add_argument("--width", type=int, default=None, help="Output width (default: scale source to long side 1280).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Multi-process inference is not supported; run this script with python.")
    if os.path.isdir(args.input):
        raise ValueError("--input must be a single video file.")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = "cuda:0"
    frames, fps = load_video(args.input, args.height, args.width)
    if not frames:
        raise ValueError(f"Input video has no decodable frames: {args.input}")

    # ---- Stage 1: translate first frames with FLUX ----
    pipe = load_flux(device, args.flux_lora)
    cutoff = args.flux_cutoff
    if args.mask is not None:
        cutoff = cutoff_map_from_mask(args.mask, frames[0].height, frames[0].width, args.mask_strength, device)
    print(f"FLUX first frame: {args.input}")
    torch.manual_seed(args.seed)
    first_frame = translate_image(
        pipe, frames[0], args.prompt, cutoff=cutoff, J=args.flux_J or None,
        lfr=not args.flux_no_lfr, negative_prompt=args.flux_negative_prompt,
    )
    del pipe
    flush()

    # ---- Stage 2: translate videos with Wan I2V ----
    pipe = load_wan(device, args.wan_high_lora, args.wan_low_lora, args.lightning_high, args.lightning_low)
    print(f"Wan video: {args.input} -> {args.output}")
    torch.manual_seed(args.seed)
    video = translate_video(
        pipe, frames, first_frame, args.prompt, cutoff=args.wan_cutoff, J=args.wan_J or None,
        lfr=not args.wan_no_lfr, negative_prompt=args.wan_negative_prompt,
    )
    save_video(video, args.output, fps=fps, quality=5)


if __name__ == "__main__":
    main()
