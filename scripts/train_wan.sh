#!/bin/bash
# ψ-PD LoRA training for one expert of Wan2.2-I2V-A14B on zengxianyu/open-sora-pexels-subset.
#
# Usage: bash scripts/train_wan.sh {high|low} [extra train_wan.py args]
#   high: high-noise expert, timesteps [900, 1000]
#   low:  low-noise expert,  timesteps [0, 900)
#
# Environment overrides:
#   NUM_GPUS   number of GPUs (default 4)
#   MIXED_PRECISION Accelerate precision mode (default no, independent of host config)
#   INIT_LORA  optional LoRA checkpoint to initialize from
#   LFR_PROB   probability of Low-Frequency Randomization (default 0.8; 0 = ψ-PD w/o LFR)
#   OUTPUT     output directory
set -e
EXPERT=$1; shift || true
case "$EXPERT" in
  high) MIN_BOUNDARY=0;     MAX_BOUNDARY=0.358 ;;
  low)  MIN_BOUNDARY=0.358; MAX_BOUNDARY=1 ;;
  *) echo "Usage: bash scripts/train_wan.sh {high|low} [args]"; exit 1 ;;
esac
NUM_GPUS=${NUM_GPUS:-4}
LFR_PROB=${LFR_PROB:-0.8}
OUTPUT=${OUTPUT:-./models/train/Wan2.2-I2V-A14B_${EXPERT}_wpd}
MIXED_PRECISION=${MIXED_PRECISION:-no}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
MULTI_GPU=""; if [ "$NUM_GPUS" -gt 1 ]; then MULTI_GPU="--multi_gpu"; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

accelerate launch --mixed_precision "$MIXED_PRECISION" $MULTI_GPU --num_processes "$NUM_GPUS" train/train_wan.py \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:${EXPERT}_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --height 704 \
  --width 1280 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64 \
  ${INIT_LORA:+--lora_checkpoint "$INIT_LORA"} \
  --extra_inputs "input_image" \
  --min_timestep_boundary "$MIN_BOUNDARY" \
  --max_timestep_boundary "$MAX_BOUNDARY" \
  --save_steps 100 \
  --use_gradient_checkpointing_offload \
  --lfr_prob "$LFR_PROB" \
  "$@"
