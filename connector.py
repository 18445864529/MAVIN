import argparse
import logging
import inspect
import os
import random

import torchvision
from glob import glob

import math
import torch.utils.checkpoint

from typing import Dict, Tuple
from omegaconf import DictConfig, OmegaConf
from einops import repeat

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import export_to_video

from mavin.data.dataset import ConnectionDataset, MAVINDataset
from mavin.utils.train_utils import *


def r(x):
    return (x + 1) * 0.5


def connection_prediction_step(dir1, dir2, config, unet, text_encoder, processor, vision_encoder, vae, pipeline, output_dir, global_step, device):
    unet.eval()
    unet_and_clip_gradient_checkpointing(unet, text_encoder, vision_encoder, enable=False)
    l_refine_v1 = config.validation_data.get('f_refine_v1', 0)
    l_refine_v2 = config.validation_data.get('f_refine_v2', 0)
    pred_length = l_refine_v1 + l_refine_v2

    stats = {'dir1': {'video_names': [], 'video_values': [], 'latent_values': [], 'image_values': []},
             'dir2': {'video_names': [], 'video_values': [], 'latent_values': [], 'image_values': []}}
    for i, dir in enumerate([dir1, dir2]):
        for vid_path in sorted(glob(f"{dir}/*.mp4")):
            video_name = vid_path.split('/')[-1].split('.')[0]
            vid = encode_video(vid_path, height=config.validation_data.height, width=config.validation_data.width, normalize=True, channel_first=True)
            stats[f'dir{i + 1}']['video_names'].append(video_name)
            stats[f'dir{i + 1}']['video_values'].append(vid)
            latent = tensor_to_vae_latent(vid.unsqueeze(0).to(vae.device, dtype=torch.float16), vae)
            stats[f'dir{i + 1}']['latent_values'].append(latent)
            if i == 0:
                img = processor(images=r(vid[- l_refine_v1 - 1]), return_tensors="pt", do_rescale=False).pixel_values[0].unsqueeze(0)
            else:
                img = processor(images=r(vid[l_refine_v2]), return_tensors="pt", do_rescale=False).pixel_values[0].unsqueeze(0)
            boundary = vision_encoder(img.to(latent.device, dtype=latent.dtype))[0]
            stats[f'dir{i + 1}']['image_values'].append(boundary)
    num_dir1_videos = len(stats['dir1']['video_names'])
    num_dir2_videos = len(stats['dir2']['video_names'])
    prompt = config.validation_data.prompt
    output_dir = config.inference_output_dir
    overwrite = config.validation_data.get('overwrite', False)

    for i in range(num_dir1_videos):
        for j in range(num_dir2_videos):
            if stats['dir1']['video_names'][i] == stats['dir2']['video_names'][j]:
                continue
            save_name = f"connected_{global_step}_{stats['dir1']['video_names'][i]}_{stats['dir2']['video_names'][j]}"
            out_file = f"{output_dir}/{save_name}.mp4"
            if os.path.exists(out_file) and not overwrite:
                print(f"Skipping {out_file}")
                continue

            boundaries = torch.cat([stats['dir1']['image_values'][i], stats['dir2']['image_values'][j]], dim=1)
            length_info = torch.tensor([pred_length], device=boundaries.device, dtype=boundaries.dtype)
            latent_v1 = stats['dir1']['latent_values'][i]
            latent_v2 = stats['dir2']['latent_values'][j]
            latent = torch.cat([latent_v1, latent_v2], dim=2)
            config.validation_data.num_frames = latent.shape[2]
            len_v1 = latent_v1.shape[2]
            p_start = len_v1 - l_refine_v1
            p_end = len_v1 + l_refine_v2
            if config.validation_data.shared_noise:
                starter = latent[:, :, p_start - 1:p_start, :, :]
                ender = latent[:, :, p_end:p_end + 1, :, :]
                if config.validation_data.get('from_boundary', False):
                    latent[:, :, p_start:p_end, :, :] = torch.cat([starter.repeat(1, 1, pred_length // 2, 1, 1), ender.repeat(1, 1, pred_length // 2, 1, 1)], dim=2)

            with torch.no_grad():
                if config.seed is not None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(config.seed)
                else:
                    generator = None

                if config.frameinit_kwargs.enable:
                    pipeline.init_filter(
                        filter_shape=latent.shape,
                        filter_params=config.frameinit_kwargs.filter_params,
                    )

                video_frames = pipeline(prompt,
                                        width=config.validation_data.width,
                                        height=config.validation_data.height,
                                        num_frames=config.validation_data.num_frames,
                                        num_inference_steps=config.validation_data.num_inference_steps,
                                        guidance_scale=config.validation_data.guidance_scale,
                                        generator=generator,
                                        starter_latent=latent,
                                        encoder_image_states=boundaries,
                                        shared_noise=config.validation_data.shared_noise,
                                        length_info=length_info,
                                        prediction_start=p_start,
                                        prediction_end=p_end,
                                        use_frameinit=config.frameinit_kwargs.enable,
                                        use_gfm=config.validation_data.use_gfm,
                                        noise_level=config.frameinit_kwargs.noise_level if config.frameinit_kwargs.enable else 999,
                                        ).frames
            os.makedirs(output_dir, exist_ok=True)
            export_to_video(video_frames, out_file, 8)
            logger.info(f"Saved a new sample to {out_file}")
            torch.cuda.empty_cache()


def train(
        # model and data
        pretrained_model_path: str,
        validation_data: DictConfig,
        inference_output_dir,
        mixed_precision: str = "fp16",
        gradient_checkpointing: bool = True,
        seed: int = None,
        resume_from_checkpoint: str = None,
        user_model_config: DictConfig = None,
        frameinit_kwargs=None,
        **kwargs
):
    if seed is not None:
        set_seed(seed)

    assert resume_from_checkpoint
    user_model_config['use_safetensors'] = True
    load_model_path = resume_from_checkpoint
    output_dir = '/'.join(resume_from_checkpoint.split('/')[:-2])
    args_info = inspect.getargvalues(inspect.currentframe())
    config = OmegaConf.create({arg: args_info.locals.get(arg) for arg in args_info.args})

    accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=1, log_with=None, project_dir=output_dir)
    create_logging(logging, logger, accelerator)
    accelerate_set_verbose(accelerator)

    # Load models and unfreeze trainable modules.
    noise_scheduler, tokenizer, text_encoder, processor, vision_encoder, vae, unet = load_primary_models(load_model_path, OmegaConf.to_container(user_model_config, resolve=True))
    freeze_models([text_encoder, vision_encoder, vae, unet])

    unet_and_clip_gradient_checkpointing(unet, text_encoder, vision_encoder, enable=gradient_checkpointing)
    vae.enable_slicing()

    # Prepare everything with our `accelerator`.
    text_encoder, vision_encoder, vae, unet = accelerator.prepare(text_encoder, vision_encoder, vae, unet)

    # Pipeline for validation.
    pipeline = MAVINPipeline.from_pretrained(pretrained_model_path, text_encoder=text_encoder, vae=vae, unet=unet)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    dir1, dir2 = validation_data.dir1, validation_data.dir2

    global_step = int(resume_from_checkpoint.split("-")[-1])

    with accelerator.autocast():
        connection_prediction_step(dir1, dir2, config, unet, text_encoder, processor, vision_encoder, vae, pipeline, output_dir, global_step, accelerator.device)


def str2bool(v):
    return v.lower() in ("y", "yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./mavin/configs/app_connection.yaml")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if config.get('keep_config', False):
        output_dir = '/'.join(config.resume_from_checkpoint.split('/')[:-2])
        config2keep = OmegaConf.load(f"{output_dir}/config.yaml")
        config2keep.resume_from_checkpoint = config.resume_from_checkpoint
        config2keep.validation_data = config.validation_data
        config2keep.frameinit_kwargs = config.frameinit_kwargs
        config2keep.inference_only = True
        config2keep.inference_output_dir = config.inference_output_dir
        config = config2keep
    train(**config)
