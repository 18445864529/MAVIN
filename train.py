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
from pytorch_msssim import ssim, ms_ssim
from eval_outputs import metric_calculation


def r(x):
    return (x + 1) * 0.5


def train_step(batch, config, unet, text_encoder, vision_encoder, vae, noise_scheduler, processor):
    unet.train()
    # Convert videos to latent space
    video_values = batch["video_values"]  # b f h w c
    prompt_ids = batch["prompt_ids"]
    latents = video_values if config.cache_latents else tensor_to_vae_latent(video_values, vae)

    bsz, video_length = latents.shape[0], latents.shape[2]

    # Encode text embeddings
    encoder_hidden_states = text_encoder(prompt_ids)[0]

    # Sample a random timestep for each video
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Variable-length sampling
    if config.train_data.get('random_mask', False):
        p_start = random.randint(config.train_data.prediction_start - 5, config.train_data.prediction_start + 5)
        p_end = random.randint(config.train_data.prediction_end - 5, config.train_data.prediction_end + 5)
        # p_start, p_end = shifted_Irwin_Hall(global_step)
    else:
        p_start, p_end = config.train_data.prediction_start, config.train_data.prediction_end

    cross_attention_kwargs = {
        "prediction_start": p_start,
        "prediction_end": p_end,
    }

    # Boundary frame guidance
    if config.get('boundary_guidance', False):
        starter_image = processor(images=r(video_values[:, p_start - 1]), return_tensors="pt", do_rescale=False).pixel_values[0].unsqueeze(0)
        boundary_s = vision_encoder(starter_image.to(vision_encoder.device, dtype=vision_encoder.dtype))[0]
        ender_image = processor(images=r(video_values[:, p_end]), return_tensors="pt", do_rescale=False).pixel_values[0].unsqueeze(0)
        boundary_e = vision_encoder(ender_image.to(vision_encoder.device, dtype=vision_encoder.dtype))[0]
        encoder_image_states = torch.cat([boundary_s, boundary_e], dim=1)
    else:
        encoder_image_states = None

    # Length embedding
    if config.train_data.get('length_emb', False):
        length_info = torch.tensor([p_end - p_start], device=latents.device, dtype=latents.dtype)
    else:
        length_info = None

    # Segment the latents into preceding, training, and following clips
    head_clip = latents[:, :, :p_start, :, :]
    tail_clip = latents[:, :, p_end:, :, :]
    training_clip = latents[:, :, p_start:p_end, :, :]

    # Sample noise and add to training clip
    use_offset_noise = config.use_offset_noise and not config.rescale_schedule
    noise = sample_noise(training_clip, config.offset_noise_strength, use_offset_noise)
    noisy_training_latent = noise_scheduler.add_noise(training_clip, noise, timesteps)
    noisy_latents = torch.cat([head_clip, noisy_training_latent, tail_clip], dim=2)
    target = noise

    # Noise prediction and loss calculation on the training clip
    model_pred = unet(noisy_latents, timesteps, length_info=length_info,
                      encoder_hidden_states=encoder_hidden_states, encoder_image_states=encoder_image_states,
                      cross_attention_kwargs=cross_attention_kwargs).sample
    loss = F.mse_loss(model_pred[:, :, p_start:p_end, :, :].float(), target.float(), reduction="mean")

    return loss, latents, p_start, p_end


def connection_prediction_step(dataset, config, unet, text_encoder, processor, vision_encoder, vae, pipeline, output_dir, global_step, device):
    unet.eval()
    unet_and_clip_gradient_checkpointing(unet, text_encoder, vision_encoder, enable=False)
    p_start = config.validation_data.prediction_start
    p_end = config.validation_data.prediction_end
    pred_length = p_end - p_start
    ckargs = {}

    latent_list, img_list, length_infos = [], [], []
    video_names = []

    # Testing data stats construction
    for vid_path in sorted(glob(f"{dataset}/*.mp4")):
        video_name = vid_path.split('/')[-1].split('.')[0]
        vid = encode_video(vid_path, height=config.validation_data.height, width=config.validation_data.width, normalize=True, channel_first=True)
        # Get video latent
        latent = tensor_to_vae_latent(vid.unsqueeze(0).to(vae.device, dtype=torch.float16), vae)
        # Replace the prediction clip with the boundary frames
        starter = latent[:, :, p_start - 1:p_start, :, :]
        ender = latent[:, :, p_end:p_end + 1, :, :]
        latent[:, :, p_start:p_end, :, :] = torch.cat([starter.repeat(1, 1, pred_length // 2, 1, 1), ender.repeat(1, 1, pred_length // 2, 1, 1)], dim=2)

        length_info = torch.tensor([p_end - p_start], device=latent.device, dtype=latent.dtype) if config.train_data.get('length_emb', False) else None

        if config.boundary_guidance:
            starter_image = processor(images=r(vid[p_start - 1]), return_tensors="pt", do_rescale=False).pixel_values[0].unsqueeze(0)
            ender_image = processor(images=r(vid[p_end]), return_tensors="pt", do_rescale=False).pixel_values[0].unsqueeze(0)
            img1 = vision_encoder(starter_image.to(vision_encoder.device, dtype=vision_encoder.dtype))[0]
            img2 = vision_encoder(ender_image.to(vision_encoder.device, dtype=vision_encoder.dtype))[0]
            img = torch.cat([img1, img2], dim=1)
        else:
            img = None

        latent_list.append(latent)
        img_list.append(img)
        length_infos.append(length_info)
        video_names.append(video_name)

    prompt = config.train_data.prompt
    output_dir = output_dir if not config.inference_only else config.inference_output_dir
    real_output_dir = os.path.join(output_dir, 'samples') if not config.inference_only else output_dir
    overwrite = False

    # Inference
    for i, (latents, encoder_image_states, l, v_name) in enumerate(zip(latent_list, img_list, length_infos, video_names)):
        config.validation_data.num_frames = latents.shape[2]
        save_name = f"{prompt}_{global_step}_{v_name}"
        out_file = f"{real_output_dir}/{save_name}.mp4"
        if os.path.exists(out_file) and not overwrite:
            continue

        with torch.no_grad():
            if config.seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(config.seed)
            else:
                generator = None

            if config.frameinit_kwargs.enable:
                pipeline.init_filter(
                    filter_shape=latents.shape,
                    filter_params=config.frameinit_kwargs.filter_params,
                )

            video_frames = pipeline(prompt,
                                    width=config.validation_data.width,
                                    height=config.validation_data.height,
                                    num_frames=config.validation_data.num_frames,
                                    num_inference_steps=config.validation_data.num_inference_steps,
                                    guidance_scale=config.validation_data.guidance_scale,
                                    generator=generator,
                                    cross_attention_kwargs=ckargs,
                                    starter_latent=latents,
                                    encoder_image_states=encoder_image_states,
                                    shared_noise=config.validation_data.shared_noise or config.validation_data.use_gfm,
                                    length_info=l,
                                    prediction_start=p_start,
                                    prediction_end=p_end,
                                    use_frameinit=config.frameinit_kwargs.enable or config.validation_data.use_gfm,
                                    use_gfm=config.validation_data.use_gfm,
                                    noise_level=config.frameinit_kwargs.noise_level,
                                    ).frames
        os.makedirs(real_output_dir, exist_ok=True)
        export_to_video(video_frames, out_file, 8)
        logger.info(f"Saved a new sample to {out_file}")
        torch.cuda.empty_cache()

    metric_calculation(real_output_dir, dataset, p_end - p_start, global_step, config.validation_data.width, processor=processor, encoder=vision_encoder, write='a')

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

        # noise
        use_offset_noise: bool = False,
        rescale_schedule: bool = False,
        offset_noise_strength: float = 0.1,

        # mavin
        user_model_config: DictConfig = None,  # additional config applied to UNet, passed from config.yaml
        frameinit_kwargs: DictConfig = None,
        skip1valid: bool = True,
        test_training: bool = False,
        boundary_guidance: bool = False,
        config: DictConfig = None,
        **kwargs
):
    # Initialization.
    if trainable_modules != 'all':
        if (train_data.get('length_emb', False)) and 'cond_proj+time_emb_proj' not in trainable_modules:
            trainable_modules += '+cond_proj+time_emb_proj+linear_1+linear_2'
            print(f"Added 'cond_proj+time_emb_proj+linear_1+linear_2' to trainable modules.")
        if user_model_config.get('augmented_attn_field', False) and 'augment_coefficient' not in trainable_modules:
            trainable_modules += '+augment_coefficient'
            print(f"Added 'augment_coefficient' to trainable modules.")

    if seed is not None:
        set_seed(seed)

    if resume_from_checkpoint:
        user_model_config['use_safetensors'] = True
        load_model_path = resume_from_checkpoint
        output_dir = '/'.join(resume_from_checkpoint.split('/')[:-2])
        args_info = inspect.getargvalues(inspect.currentframe())
        config = config or OmegaConf.create({arg: args_info.locals.get(arg) for arg in args_info.args})

    else:
        load_model_path = pretrained_model_path
        args_info = inspect.getargvalues(inspect.currentframe())
        config = config or OmegaConf.create({arg: args_info.locals.get(arg) for arg in args_info.args})
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

    unet_and_clip_gradient_checkpointing(unet, text_encoder, vision_encoder, enable=gradient_checkpointing)
    vae.enable_slicing()

    # Optimizer and scheduler.
    learning_rate *= (learning_rate * gradient_accum * train_batch_size * accelerator.num_processes) if scale_lr else 1
    adam_beta1, adam_beta2, adam_weight_decay, adam_epsilon = adam_args
    params = create_optimizer_params(unet, weight_decay=adam_weight_decay)
    optimizer_cls = get_optimizer(use_8bit_adam)
    optimizer = optimizer_cls(params, lr=learning_rate, betas=(adam_beta1, adam_beta2), eps=adam_epsilon)
    lr_scheduler = get_scheduler(lr_scheduler, optimizer, num_warmup_steps=lr_warmup_steps * gradient_accum, num_training_steps=max_train_steps * gradient_accum)

    # Prepare everything with our `accelerator`.
    text_encoder, vision_encoder, vae, unet, optimizer, lr_scheduler = \
        accelerator.prepare(text_encoder, vision_encoder, vae, unet, optimizer, lr_scheduler)

    # Pipeline for validation.
    pipeline = MAVINPipeline.from_pretrained(pretrained_model_path, text_encoder=text_encoder, vae=vae, unet=unet)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    global_step = 0
    skip_after_resume = False

    if resume_from_checkpoint:
        global_step = int(resume_from_checkpoint.split("-")[-1])
        skip_after_resume = True

    if inference_only:
        with accelerator.autocast():
            connection_prediction_step(validation_data.test_root, config, unet, text_encoder, processor, vision_encoder, vae, pipeline, output_dir, global_step, accelerator.device)
            import sys
            sys.exit(0)

    # Dataset and Dataloader.
    train_dataset = ConnectionDataset(tokenizer, processor, **train_data)
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

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    progress_bar.update(global_step)

    if test_training:
        unet.requires_grad_(False)

    # Train!
    for epoch in range(num_train_epochs):
        end_training = False
        if end_training:
            break

        for _, batch in enumerate(train_dataloader):

            # Do validation on the test sample and save if checkpoints.
            if not test_training and global_step % validation_steps == 0 and accelerator.sync_gradients:
                if (global_step == 0 and skip1valid) or (skip_after_resume and skip1valid):
                    skip_after_resume = False
                    pass
                else:
                    if global_step % checkpointing_steps == 0 and global_step > 0:  # expected behavior
                        save_pipe(pretrained_model_path, global_step, accelerator, unet, text_encoder, vae, output_dir,
                                  is_checkpoint=True, save_pretrained_model=save_pretrained_model)
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            connection_prediction_step(validation_data.test_root, config, unet, text_encoder, processor, vision_encoder, vae, pipeline, output_dir, global_step, accelerator.device)

            if global_step >= max_train_steps:
                end_training = True
                break

            # Do a training step.
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    loss, latents, s, e = train_step(batch, config, unet, text_encoder, vision_encoder, vae, noise_scheduler, processor)
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

                logs = {"step_loss": ddp_loss, "s": s, "e": e}
                progress_bar.set_postfix(**logs)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def str2bool(v):
    return v.lower() in ("y", "yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./mavin/configs/connection.yaml")
    # parser.add_argument("-c", "--config", type=str, default="./mavin/configs/inference.yaml")
    parser.add_argument("-d", "--debug", type=str2bool, default='no', help="Dummy output and validation for debugging")
    parser.add_argument("-s", "--skip1valid", type=str2bool, default='yes', help="skip the very first validation before training")
    parser.add_argument("-t", "--test_training", type=str2bool, default='no', help="test training")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if config.get('keep_config', False):
        output_dir = '/'.join(config.resume_from_checkpoint.split('/')[:-2])
        config2keep = OmegaConf.load(f"{output_dir}/config.yaml")
        config2keep.resume_from_checkpoint = config.resume_from_checkpoint
        config2keep.max_train_steps = config.max_train_steps
        config2keep.validation_data = config.validation_data
        config2keep.frameinit_kwargs = config.frameinit_kwargs
        if config.get('inference_only', False):
            config2keep.inference_only = True
            config2keep.inference_output_dir = config.inference_output_dir

        # add those that do not exist in the original config due to version update
        for k, v in config.items():
            if k not in config2keep:
                config2keep[k] = v

        config = config2keep
        do_not_create_config = config
    else:
        do_not_create_config = None

    config.debug = args.debug
    config.skip1valid = args.skip1valid
    config.test_training = args.test_training
    train(config=do_not_create_config, **config)
