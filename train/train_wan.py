"""ψ-PD LoRA training for Wan2.2-I2V-A14B (one expert at a time) on unpaired videos.
See scripts/train_wan_high.sh and scripts/train_wan_low.sh."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffsynth.pipelines.wan_video_new import WanVideoPipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from train.common import launch_training, log_step, sample_training_noise
from train.data import create_video_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(self, model_paths=None, model_id_with_origin_paths=None, trainable_models=None,
                 lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
                 lora_checkpoint=None, use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
                 extra_inputs=None, max_timestep_boundary=1.0, min_timestep_boundary=0.0, lfr_prob=0.8):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models, lora_base_model, lora_target_modules, lora_rank,
            lora_checkpoint=lora_checkpoint, enable_fp8_training=False,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.lfr_prob = lfr_prob
        self.accelerator = None

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator
        self.pipe.device = accelerator.device
        self.pipe.vae_device = accelerator.device
        self.pipe.enable_cpu_offload()

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": True,
            "tile_size": (30, 52),
            "tile_stride": (15, 26),
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        # I2V: condition on the first frame.
        if "input_image" in self.extra_inputs:
            inputs_shared["input_image"] = data["video"][0]
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, step=None):
        with torch.no_grad():
            inputs = self.forward_preprocess(data)
        self.pipe.load_models_to_device(self.pipe.in_iteration_models)

        # Per-meta-frame noise: (1, C, T, H, W) -> (T, C, H, W) -> back.
        frames = inputs["input_latents"][0].transpose(0, 1).float()
        noise, stats = sample_training_noise(frames, self.lfr_prob)
        noise = noise.transpose(0, 1)[None].contiguous()
        inputs["noise"] = noise.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        if self.accelerator is not None and step is not None:
            log_step(self.accelerator, noise, loss, stats, step)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--hf_dataset", type=str, default="zengxianyu/open-sora-pexels-subset",
                        help="HF dataset repo with videos/*.tar and annotations/captions.parquet.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit videos for short/debug runs; None uses the full dataset.")
    parser.add_argument("--no_streaming", action="store_true",
                        help="Download the full dataset first instead of streaming it.")
    parser.add_argument("--lfr_prob", type=float, default=0.8,
                        help="Probability of Low-Frequency Randomization per step (0 = ψ-PD w/o LFR).")
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
    args = parser.parse_args()

    streaming = not args.no_streaming
    dataset = create_video_dataset(
        args.hf_dataset, num_frames=args.num_frames, height=args.height, width=args.width,
        max_pixels=args.max_pixels, streaming=streaming, max_samples=args.max_samples,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        lfr_prob=args.lfr_prob,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launch_training(dataset, model, model_logger, args, shuffle=not streaming)
