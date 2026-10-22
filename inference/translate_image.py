"""
ψ-PD image sim-to-real translation with FLUX.1-dev.

Translate a single image:
    python inference/translate_image.py --input sim.png --output real.png

Instance-level translation (white mask pixels are translated, the rest stays untranslated):
    python inference/translate_image.py --input sim.png --mask mask.png --no_lfr --output real.png
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wpd.pipelines import (FLUX_NEGATIVE_PROMPT, cutoff_map_from_mask,
                           load_flux, load_image, translate_image)

DEFAULT_PROMPT = (
    "A photorealistic photograph taken from a forward-facing vehicle-mounted camera. "
    "Natural outdoor lighting, authentic surface textures, real-world colors."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--mask", default=None, help="Instance mask (white = translate).")
    parser.add_argument("--mask_strength", type=float, default=0.5,
                        help="Cutoff reduction inside the mask: cutoff = 1 - strength (higher = stronger translation).")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default=FLUX_NEGATIVE_PROMPT)
    parser.add_argument("--cutoff", type=float, default=0.5,
                        help="Global cutoff as a fraction of the latent Nyquist frequency (ignored with --mask).")
    parser.add_argument("--J", type=int, default=4, help="DT-CWPT decomposition depth (0 = auto).")
    parser.add_argument("--no_lfr", action="store_true", help="Disable Low-Frequency Randomization.")
    parser.add_argument("--lora", default="checkpoints/flux.safetensors")
    parser.add_argument("--height", type=int, default=None, help="Output height (default: scale source to long side 1280).")
    parser.add_argument("--width", type=int, default=None, help="Output width (default: scale source to long side 1280).")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Multi-process inference is not supported; run this script with python.")
    if os.path.isdir(args.input):
        raise ValueError("--input must be a single image file.")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = "cuda:0"
    image = load_image(args.input, args.height, args.width)

    pipe = load_flux(device, args.lora)
    cutoff = args.cutoff
    if args.mask is not None:
        cutoff = cutoff_map_from_mask(args.mask, image.height, image.width, args.mask_strength, device)

    print(f"Translating {args.input} -> {args.output}")
    torch.manual_seed(args.seed)
    result = translate_image(
        pipe, image, args.prompt,
        cutoff=cutoff, J=args.J or None, lfr=not args.no_lfr,
        negative_prompt=args.negative_prompt, cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
    )
    result.save(args.output)


if __name__ == "__main__":
    main()
