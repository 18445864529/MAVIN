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

from mavin.data.dataset import MAVINDataset
from mavin.utils.train_utils import *


def train_step(batch, config, unet, text_encoder, vision_encoder, vae, noise_scheduler, processor, raft):
    unet.train()
    # Convert videos to latent space
    video_values = batch["video_values"]  # b f h w c
    image_values = batch["image_values"]
    prompt_ids = batch["prompt_ids"]
    condition_frame_index = batch["condition_frame_index"]
    latents = video_values if config.cache_latents else tensor_to_vae_latent(video_values, vae)
    condition_frame_latents = latents[:, :, condition_frame_index, :, :]  # b c 1 h w

    bsz, video_length = latents.shape[0], latents.shape[2]
    # latent_motion_value = calculate_latent_motion_score(latents) if config.use_motion_loss else None

    # Encode text embeddings
    encoder_hidden_states = text_encoder(prompt_ids)[0]

    # Encode image embeddings for extra conditioning
    encoder_image_states = vision_encoder(image_values[:, -1])[0] if unet.user_model_config.get("cross_image_condition", False) else None  # 1, 257, 1280

    # Sample noise that we'll add to the latents
    use_offset_noise = config.use_offset_noise and not config.rescale_schedule

    # Sample a random timestep for each video
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    noise = sample_noise(latents, config.offset_noise_strength, use_offset_noise)
    # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    assert video_length == noisy_latents.shape[2] > 1, "Method is not implement for static images, video length must be greater than 1."

    # Get the target for loss depending on the prediction type
    if noise_scheduler.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

    if config.image_conditioning:
        noisy_latents[:, :, :1, :, :] = latents[:, :, :1, :, :] * noise_scheduler.init_noise_sigma  # b c f h w

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, encoder_image_states=encoder_image_states).sample
        loss = F.mse_loss(model_pred[:, :, 1:, :, :].float(), target[:, :, 1:, :, :].float(), reduction="mean")
        # loss += 3e-5 * F.mse_loss(latent_motion_value, calculate_latent_motion_score(model_pred)) if config.use_motion_loss else 0

    else:
        if config.train_data.random_condition_frame:
            cross_attention_kwargs = {
                "condition_frame_latents": condition_frame_latents,
                "condition_frame_index": condition_frame_index,
                "skip_zeroth": True,
            }
            noisy_latents = torch.cat([condition_frame_latents, noisy_latents], dim=2)
        else:
            cross_attention_kwargs = None

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                          encoder_image_states=encoder_image_states, cross_attention_kwargs=cross_attention_kwargs).sample
        if config.train_data.random_condition_frame:
            loss = F.mse_loss(model_pred[:, :, 1:, :, :].float(), target.float(), reduction="mean")
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss, latents


def inference_call(config, img_spath, vae, processor, vision_encoder, unet, pipeline, out_file, prompt, device):
    with torch.no_grad():
        starter_image, starter_latent = encode_image(img_spath, vae, config.validation_data.width, config.validation_data.height)
        starter_image_values = processor(images=starter_image, return_tensors="pt").pixel_values[0].unsqueeze(0)  # only support bsz 1
        encoder_image_states = vision_encoder(starter_image_values.to(vision_encoder.device, dtype=vision_encoder.dtype))[0]
        if not unet.user_model_config.get("cross_image_condition"):
            encoder_image_states = None

        if config.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed)
        else:
            generator = None

        if config.frameinit_kwargs.enable:
            pipeline.init_filter(
                filter_shape=[starter_latent.shape[0], starter_latent.shape[1], config.validation_data.num_frames, starter_latent.shape[3], starter_latent.shape[4]],
                filter_params=config.frameinit_kwargs.filter_params,
            )

        video_frames = pipeline(prompt,
                                width=config.validation_data.width,
                                height=config.validation_data.height,
                                num_frames=config.validation_data.num_frames,
                                num_inference_steps=config.validation_data.num_inference_steps,
                                guidance_scale=config.validation_data.guidance_scale,
                                generator=generator,
                                starter_latent=starter_latent,
                                encoder_image_states=encoder_image_states,
                                image_conditioning=config.image_conditioning,
                                shared_noise=config.validation_data.shared_noise,
                                use_frameinit=config.frameinit_kwargs.enable,
                                sc_latent=config.train_data.get('random_condition_frame', False),
                                ).frames
    export_to_video(video_frames, out_file, 8)
    logger.info(f"Saved a new sample to {out_file}")
    torch.cuda.empty_cache()


def validate_step(config, unet, text_encoder, processor, vision_encoder, vae, pipeline, output_dir, global_step, device):
    unet.eval()
    unet_and_clip_gradient_checkpointing(unet, text_encoder, vision_encoder, enable=False)

    prompt = config.train_data.prompt if len(config.validation_data.prompt) <= 0 else config.validation_data.prompt

    output_dir = output_dir if not config.inference_only else config.inference_output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if config.validation_data.get('data_dir', None):
        paths = glob(config.validation_data.data_dir + "/*.png")
    else:
        paths = config.validation_data.starter_image_path
        if isinstance(paths, str):
            paths = [paths]

    for i, starter_image_path in enumerate(paths):
        if prompt == 'rjes':
            prompts = config.train_data.prompt
        else:
            prompts = [prompt]

        for p in prompts:
            save_name = f"{global_step}_{i}_{p}"
            out_file = f"{output_dir}/samples/{save_name}.mp4" if not config.inference_only else f"{output_dir}/{save_name}.mp4"
            inference_call(config, starter_image_path, vae, processor, vision_encoder, unet, pipeline, out_file, p, device)

    unet_and_clip_gradient_checkpointing(unet, text_encoder, vision_encoder, enable=config.gradient_checkpointing)


def train(
        # model and data
        pretrained_model_path: str,
        output_root: str,
        experiment_name: str,
        train_data: DictConfig,
        validation_data: DictConfig,
        shuffle: bool = True,
        trainable_modules: Tuple[str] = None,  # Eg: ("attn1", "attn2")
        debug: bool = False,

        # optimizer
        adam_args: tuple = (0.9, 0.999, 1e-2, 1e-08),  # beta1, beta2, weight_decay, eps
        learning_rate: float = 5e-5,
        lr_scheduler: str = "constant",
        scale_lr: bool = False,
        lr_warmup_steps: int = 0,
        gradient_accum: int = 1,
        max_grad_norm: float = 1.0,

        # training and validation
        train_batch_size: int = 1,
        max_train_steps: int = 500,
        validation_steps: int = 500,
        checkpointing_steps: int = 500,
        inference_only: bool = False,
        inference_output_dir: str = './mavin/inference_outputs',

        # speed-up
        mixed_precision: str = "fp16",
        use_8bit_adam: bool = False,
        gradient_checkpointing: bool = True,
        cache_latents: bool = False,
        cached_latent_dir=None,

        # pipeline
        seed: int = None,
        resume_from_checkpoint: str = None,
        save_pretrained_model: bool = True,
        logger_type: str = None,

        # others
        use_offset_noise: bool = False,
        rescale_schedule: bool = False,
        offset_noise_strength: float = 0.1,

        # mavin
        image_conditioning: bool = False,
        user_model_config: DictConfig = None,  # additional config applied to UNet, passed from config.yaml
        connection_prediction: bool = False,
        use_motion_loss: bool = False,
        frameinit_kwargs: DictConfig = None,
        skip1valid: bool = True,
        test_training: bool = False,
        keep_config: bool = False,
):
    # Initialization.
    if resume_from_checkpoint:
        user_model_config['use_safetensors'] = True

    if seed is not None:
        set_seed(seed)
    if resume_from_checkpoint:
        load_model_path = resume_from_checkpoint
        output_dir = '/'.join(resume_from_checkpoint.split('/')[:-2])
        if keep_config:
            config = OmegaConf.load(f"{output_dir}/config.yaml")
            for k, v in config.items():
                locals()[k] = v


        else:
            args_info = inspect.getargvalues(inspect.currentframe())
            config = OmegaConf.create({arg: args_info.locals.get(arg) for arg in args_info.args})
        if inference_only:
            output_dir = inference_output_dir
            config.inference_only = True
            config.inference_output_dir = output_dir
            config.validation_data = validation_data
    else:
        load_model_path = pretrained_model_path
        args_info = inspect.getargvalues(inspect.currentframe())
        config = OmegaConf.create({arg: args_info.locals.get(arg) for arg in args_info.args})
        output_dir = make_exp_dir_and_save_config(output_root, experiment_name, config) if not debug else None
    trainable_modules = parse_trainable_modules(trainable_modules)
    logger_type = logger_type if not debug else None
    accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=gradient_accum, log_with=logger_type, project_dir=output_dir)
    create_logging(logging, logger, accelerator)
    accelerate_set_verbose(accelerator)
    if accelerator.is_main_process and not debug:
        accelerator.init_trackers("mavin_tracker")

    # Load models and unfreeze trainable modules.
    noise_scheduler, tokenizer, text_encoder, processor, vision_encoder, vae, unet = load_primary_models(load_model_path, OmegaConf.to_container(user_model_config, resolve=True))
    freeze_models([text_encoder, vision_encoder, vae, unet])
    handle_trainable_modules(unet, trainable_modules)
    if rescale_schedule and not use_offset_noise:
        noise_scheduler.betas = enforce_zero_terminal_snr(noise_scheduler.betas)

    # Memory saver.
    unet_and_clip_gradient_checkpointing(unet, text_encoder, vision_encoder, enable=gradient_checkpointing)
    vae.enable_slicing()

    # Optimizer and scheduler.
    learning_rate *= (learning_rate * gradient_accum * train_batch_size * accelerator.num_processes) if scale_lr else 1
    adam_beta1, adam_beta2, adam_weight_decay, adam_epsilon = adam_args
    params = create_optimizer_params(unet, weight_decay=adam_weight_decay)
    optimizer_cls = get_optimizer(use_8bit_adam)
    optimizer = optimizer_cls(params, lr=learning_rate, betas=(adam_beta1, adam_beta2), eps=adam_epsilon)
    lr_scheduler = get_scheduler(lr_scheduler, optimizer, num_warmup_steps=lr_warmup_steps * gradient_accum, num_training_steps=max_train_steps * gradient_accum)
    raft = torchvision.models.optical_flow.raft_large(pretrained=True)

    # Prepare everything with our `accelerator`.
    text_encoder, vision_encoder, vae, unet, optimizer, lr_scheduler, raft = \
        accelerator.prepare(text_encoder, vision_encoder, vae, unet, optimizer, lr_scheduler, raft)

    # Pipeline for validation.
    pipeline = MAVINPipeline.from_pretrained(pretrained_model_path, text_encoder=text_encoder, vae=vae, unet=unet)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


    global_step = 0
    skip_after_resume = False
    raft.eval()

    # Dataset and Dataloader.
    if not inference_only:
        train_dataset = MAVINDataset(tokenizer, processor, **train_data)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)
        train_dataloader = handle_cache_latents(cache_latents, cached_latent_dir, train_dataloader, train_batch_size, vae, dtype=get_weight_dtype(accelerator))
        train_dataloader = accelerator.prepare(train_dataloader)

        # Recalculate total training steps and epochs.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accum)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (with distributed & accumulation) = {train_batch_size * accelerator.num_processes * gradient_accum}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accum}")
        logger.info(f"  Total optimization steps = {max_train_steps}")

    # Resume training.
    if resume_from_checkpoint:
        global_step = int(resume_from_checkpoint.split("-")[-1])
        skip_after_resume = True

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    progress_bar.update(global_step)

    if test_training:
        unet.requires_grad_(False)

    if inference_only:
        with accelerator.autocast():
            validate_step(config, unet, text_encoder, processor, vision_encoder, vae, pipeline, output_dir, global_step, accelerator.device)
        if inference_only:
            import sys
            sys.exit(0)

    # Train!
    for epoch in range(num_train_epochs):
        end_training = False
        if end_training:
            break

        for _, batch in enumerate(train_dataloader):

            # Do validation on the test sample and save if checkpoints.
            if not test_training and global_step % validation_steps == 0 and accelerator.sync_gradients or inference_only:
                if (global_step == 0 and skip1valid) or (skip_after_resume and not inference_only and skip1valid):
                    skip_after_resume = False
                    pass
                else:
                    if global_step % checkpointing_steps == 0 and global_step > 0 and not inference_only:  # expected behavior
                        save_pipe(pretrained_model_path, global_step, accelerator, unet, text_encoder, vae, output_dir,
                                  is_checkpoint=True, save_pretrained_model=save_pretrained_model)
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            validate_step(config, unet, text_encoder, processor, vision_encoder, vae, pipeline, output_dir, global_step, accelerator.device)
                        if inference_only:
                            import sys
                            sys.exit(0)

            if global_step >= max_train_steps:
                end_training = True
                break

            # Do a training step.
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    loss, latents = train_step(batch, config, unet, text_encoder, vision_encoder, vae, noise_scheduler, processor, raft)
                if not test_training:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and max_grad_norm:
                        accelerator.clip_grad_norm_(list(unet.parameters()), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # Status update and checkpointing.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                ddp_loss = accelerator.gather(loss.repeat(train_batch_size)).mean().item()
                accelerator.log({"train_loss": ddp_loss}, step=global_step)

                logs = {"step_loss": ddp_loss, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def str2bool(v):
    return v.lower() in ("y", "yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./mavin/configs/tiger_action.yaml")
    parser.add_argument("-d", "--debug", type=str2bool, default='no', help="Dummy output and validation for debugging")
    parser.add_argument("-s", "--skip1valid", type=str2bool, default='no', help="skip the first validation before training")
    parser.add_argument("-t", "--test_training", type=str2bool, default='no', help="test training")
    args = parser.parse_args()

    train(**OmegaConf.load(args.config), debug=args.debug, skip1valid=args.skip1valid, test_training=args.test_training)
