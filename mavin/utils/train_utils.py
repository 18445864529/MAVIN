import datetime
import os
import gc
import numpy
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import diffusers
import transformers

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from einops import rearrange
from PIL import Image

from accelerate.logging import get_logger
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPVisionModel
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models import AutoencoderKL

from mavin.models.attention_processor import AttnProcessor2_0
from mavin.models.attention import BasicTransformerBlock
from mavin.data.dataset import CachedDataset
from mavin.pipeline_mavin import MAVINPipeline
from mavin.models.unet_3d_condition import UNet3DConditionModel

logger = get_logger(__name__, log_level="INFO")


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def make_exp_dir_and_save_config(output_dir, exp_name, config):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")[5:-3]
    out_dir = os.path.join(output_dir, f"{now}_{exp_name}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir


def calculate_latent_motion_score(latents):
    # latents b, c, f, h, w
    diff = torch.abs(latents[:, :, 1:] - latents[:, :, :-1])
    motion_score = torch.sum(torch.mean(diff, dim=[2, 3, 4]), dim=1) * 10
    return motion_score


def parse_trainable_modules(modules_repr):
    if not isinstance(modules_repr, str):
        return modules_repr
    if modules_repr == 'all':
        return 'all'
    module_list = []
    modules = modules_repr.split('+')
    for m in modules:
        if m == 'tempatt':
            module_list.append('temp_attentions')
        elif m == 'spaatt':
            module_list.append('.attentions')
        elif m == 'scatt':
            module_list.append(['.attentions', 'attn1'])  # should meet both conditions
        elif m == 'spacross':
            module_list.append(['.attentions', 'attn2'])
        elif m == 'allatt':
            module_list.extend(['attn1', 'attn2'])  # should meet either condition
        elif m == 'tempconv':
            module_list.append('temp_conv')
        elif m == 'adpt':
            module_list.extend(['adapter', 'image_cond_layer'])
        else:
            module_list.append(m)

    print('trainable parameters:', module_list)
    return module_list


def handle_trainable_modules(model, trainable_modules=None):
    if not trainable_modules:
        return
    if trainable_modules == 'all':
        model.requires_grad_(True)
        print(f"All layers unfrozen.")
        return

    model.requires_grad_(False)
    trainable_layer_list = []
    trainable_layers = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        for tm in trainable_modules:
            condition = 'tm in name' if not isinstance(tm, list) else " and ".join([f"'{t}' in name" for t in tm])
            if eval(condition) and name not in trainable_layer_list:
                param.requires_grad_(True)
                try:
                    trainable_params += torch.prod(torch.tensor(param.size()))
                except:
                    assert tm == 'augment_coefficient'
                trainable_layer_list.append(name)
                trainable_layers += 1
    print("####################################################")
    print(f"{trainable_layers} layers unfrozen, with {human_readable(trainable_params)} trainable parameters.")
    print("####################################################")


def load_primary_models(pretrained_model_path, user_model_config=None):
    """
    :param pretrained_model_path: refer to the desired pretrained folder structure
    :param additional_kwargs: this will override the config file if exist or add new kwargs to config if non-exist
    :return:
    """
    # clip_image_repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    clip_image_repo = "openai/clip-vit-large-patch14"
    freezed_models_path = "/data/a/bowenz/models/text-to-video-ms-1.7b"
    noise_scheduler = DDIMScheduler.from_config(freezed_models_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(freezed_models_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(freezed_models_path, subfolder="text_encoder")
    processor = CLIPProcessor.from_pretrained(clip_image_repo)
    vision_encoder = CLIPVisionModel.from_pretrained(clip_image_repo)
    vae = AutoencoderKL.from_pretrained(freezed_models_path, subfolder="vae")
    # unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet = UNet3DConditionModel.partially_from_pretrained(pretrained_model_path, subfolder="unet", user_model_config=user_model_config)
    return noise_scheduler, tokenizer, text_encoder, processor, vision_encoder, vae, unet


def unet_and_clip_gradient_checkpointing(unet, text_encoder, vision_encoder, enable=True):
    unet.enable_gradient_checkpointing() if enable else unet.disable_gradient_checkpointing()
    text_encoder._set_gradient_checkpointing(enable)
    vision_encoder._set_gradient_checkpointing(enable)


def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None:
            model.requires_grad_(False)


def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if name.split('.')[-1] in ('attn1', 'attn2'):
            if isinstance(module, torch.nn.ModuleList):

                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        for attn in [m.attn1, m.attn2]:
                            attn.set_processor(AttnProcessor2_0())
                        optim_count += 1

    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn

        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if enable_torch_2:
            set_torch_2_attn(unet)
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")


def create_optimizer_params(*trainable_models, weight_decay=0):
    params = []
    for model in trainable_models:
        p_wd, p_non_wd = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
        params.extend([{"name": "decay", "params": p_wd, "weight_decay": weight_decay}, {"name": "no_decay", "params": p_non_wd, "weight_decay": 0}])
    return params


def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW


def get_weight_dtype(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype


def cast_to_gpu_and_dtype(model_list, device='cuda', dtype=torch.float32):
    for model in model_list:
        if model is not None:
            model.to(device, dtype=dtype)


def handle_cache_latents(
        should_cache,
        cached_latent_dir,
        train_dataloader,
        train_batch_size,
        vae,
        shuffle=False,
        dtype=torch.float16
):
    # Cache latents by storing them in VRAM.
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache:
        return train_dataloader

    vae.to('cuda', dtype=dtype)
    vae.enable_slicing()

    # if not exist the dir or the dir is empty, we will cache the latents
    if not os.path.exists(cached_latent_dir) or len(os.listdir(cached_latent_dir)) == 0:
        os.makedirs(cached_latent_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            save_name = f"cached_{i}"
            full_out_path = f"{cached_latent_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to('cuda', dtype=dtype)
            batch['pixel_values'] = tensor_to_vae_latent(pixel_values, vae)
            for k, v in batch.items():
                batch[k] = v[0]

            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()

    return torch.utils.data.DataLoader(CachedDataset(cache_dir=cached_latent_dir),
                                       batch_size=train_batch_size,
                                       shuffle=shuffle,
                                       num_workers=0)


def human_readable(tensor):
    num = tensor.item()
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents


def encode_image(image_path, vae, width, height):
    image = Image.open(image_path).resize((width, height)).convert('RGB')
    image_tensor = torch.from_numpy(numpy.array(image)).permute(2, 0, 1).unsqueeze(0).to(vae.device)  # maybe consider casting to mixed precision
    image_tensor = image_tensor / 127.5 - 1.0
    latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215  # b c h w
    latent = latent.unsqueeze(2)  # b c 1 h w
    return image, latent


def encode_video(video_path, width=256, height=256, normalize=False, channel_first=False):
    import decord
    vr = decord.VideoReader(video_path, width=width, height=height)
    video = vr.get_batch(range(len(vr)))
    if channel_first:
        video = rearrange(video, "f h w c -> f c h w")
    if normalize:
        video = video / 127.5 - 1.0
    return video


def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents


def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas


def save_pipe(path, global_step, accelerator, unet, text_encoder, vae, output_dir, is_checkpoint=False, save_pretrained_model=True):
    if is_checkpoint:
        save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so that we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype

    pipeline = MAVINPipeline.from_pretrained(path, unet=unet, text_encoder=text_encoder, vae=vae).to(torch_dtype=torch.float32)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")

    torch.cuda.empty_cache()
    gc.collect()


def resume_from_checkpoint(checkpoint, unet, text_encoder, vision_encoder, vae, optimizer, lr_scheduler, max_train_steps, accelerator):
    checkpoint = torch.load(checkpoint)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    vision_encoder.load_state_dict(checkpoint['vision_encoder_state_dict'])
    vae.load_state_dict(checkpoint['vae_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    global_step = checkpoint['global_step']
    first_epoch = checkpoint['epoch']
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


def remove_noise(
        scheduler,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = scheduler.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    removed = (original_samples - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
    return removed


def decode_latents(latents, vae, for_training=False):
    latents = 1 / vae.config.scaling_factor * latents

    b = latents.shape[0]
    latents = rearrange(latents, 'b c f h w -> (b f) c h w')

    image = vae.decode(latents).sample
    if not for_training:
        image = image.detach()
    video = rearrange(image, '(b f) c h w -> b f h w c', b=b)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    video = video.float()

    mean = torch.tensor([0.5, 0.5, 0.5], device=video.device).reshape(1, 1, 1, 1, -1)
    std = torch.tensor([0.5, 0.5, 0.5], device=video.device).reshape(1, 1, 1, 1, -1)

    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    if for_training:
        return video  # b f h w c
    video.clamp_(0, 1)
    images = rearrange(video, 'b f h w c -> f h (b w) c')
    images = images.unbind(dim=0)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]

    return images
