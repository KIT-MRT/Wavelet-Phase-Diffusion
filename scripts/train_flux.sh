#!/bin/bash
# ψ-PD LoRA training for FLUX.1-dev on bghira/photo-concept-bucket.
#
# Environment overrides:
#   NUM_GPUS   number of GPUs (default 4)
#   MIXED_PRECISION Accelerate precision mode (default no, independent of host config)
#   INIT_LORA  optional LoRA checkpoint to initialize from
#   LFR_PROB   probability of Low-Frequency Randomization (default 0.8; 0 = ψ-PD w/o LFR)
#   OUTPUT     output directory
set -e
NUM_GPUS=${NUM_GPUS:-4}
LFR_PROB=${LFR_PROB:-0.8}
OUTPUT=${OUTPUT:-./models/train/FLUX.1-dev_wpd}
MIXED_PRECISION=${MIXED_PRECISION:-no}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
MULTI_GPU=""; if [ "$NUM_GPUS" -gt 1 ]; then MULTI_GPU="--multi_gpu"; fi

accelerate launch --mixed_precision "$MIXED_PRECISION" $MULTI_GPU --num_processes "$NUM_GPUS" train/train_flux.py \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --max_pixels 2073600 \
  --dataset_repeat 1 \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  ${INIT_LORA:+--lora_checkpoint "$INIT_LORA"} \
  --align_to_opensource_format \
  --save_steps 1000 \
  --use_gradient_checkpointing \
  --lfr_prob "$LFR_PROB" \
  "$@"
