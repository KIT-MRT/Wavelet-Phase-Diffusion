#!/bin/bash
# Download the ψ-PD LoRAs (FLUX + Wan2.2) and the 4-step Wan2.2-Lightning LoRAs into ./checkpoints.
# Base models (FLUX.1-dev, Wan2.2-I2V-A14B) are downloaded automatically into ./models on first use.
set -e
if command -v hf >/dev/null; then CLI="hf"; else CLI="huggingface-cli"; fi
$CLI download kaiwen-wang/Wavelet-Phase-Diffusion --local-dir checkpoints "$@"
