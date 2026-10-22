"""ψ-PD LoRA training for FLUX.1-dev on unpaired image-caption data. See scripts/train_flux.sh."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image_new import FluxImagePipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, flux_parser
from train.common import launch_training, log_step, sample_training_noise
from train.data import URLImageDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FluxTrainingModule(DiffusionTrainingModule):
    def __init__(self, model_paths=None, model_id_with_origin_paths=None, trainable_models=None,
                 lora_base_model=None, lora_target_modules=None, lora_rank=32, lora_checkpoint=None,
                 use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False, lfr_prob=0.8):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        self.mapping_lora_state_dict = FluxLoRAConverter.align_to_diffsynth_format
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models, lora_base_model, lora_target_modules, lora_rank,
            lora_checkpoint=lora_checkpoint, enable_fp8_training=False,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.lfr_prob = lfr_prob
        self.accelerator = None

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "cfg_scale": 1,
            "embedded_guidance": 1,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, step=None):
        inputs = self.forward_preprocess(data)
        noise, stats = sample_training_noise(inputs["input_latents"], self.lfr_prob)
        inputs["noise"] = noise.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        if self.accelerator is not None and step is not None:
            log_step(self.accelerator, noise, loss, stats, step)
        return loss


if __name__ == "__main__":
    parser = flux_parser()
    parser.add_argument("--dataset_name", type=str, default="bghira/photo-concept-bucket",
                        help="HF dataset with image URLs and captions.")
    parser.add_argument("--url_field", type=str, default="url")
    parser.add_argument("--text_field", type=str, default="cogvlm_caption")
    parser.add_argument("--image_cache_dir", type=str, default="./data/images")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit the dataset size (debugging).")
    parser.add_argument("--lfr_prob", type=float, default=0.8,
                        help="Probability of Low-Frequency Randomization per step (0 = ψ-PD w/o LFR).")
    args = parser.parse_args()

    dataset = URLImageDataset(
        dataset_name=args.dataset_name,
        url_field=args.url_field,
        text_field=args.text_field,
        cache_dir=args.image_cache_dir,
        max_pixels=args.max_pixels,
        repeat=args.dataset_repeat,
        max_samples=args.max_samples,
    )
    print(f"Dataset size: {len(dataset)}")
    model = FluxTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        lfr_prob=args.lfr_prob,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else lambda x: x,
    )
    launch_training(dataset, model, model_logger, args)
