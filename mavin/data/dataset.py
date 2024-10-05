import torchvision
from torch.utils.data import Dataset
from einops import rearrange
import torch
import json

import os
import decord
import numpy as np
import random
import torchvision.transforms as T
from pytorch_msssim import ssim, ms_ssim

from glob import glob
from PIL import Image
from itertools import islice
from tqdm import tqdm

decord.bridge.set_bridge('torch')


class CachedDataset(Dataset):
    def __init__(self, cache_dir: str = ''):
        self.cache_dir = cache_dir
        self.cached_data_list = self.get_files_list()

    def get_files_list(self):
        tensors_list = [f"{self.cache_dir}/{x}" for x in os.listdir(self.cache_dir) if x.endswith('.pt')]
        return sorted(tensors_list)

    def __len__(self):
        return len(self.cached_data_list)

    def __getitem__(self, index):
        cached_latent = torch.load(self.cached_data_list[index], map_location='cpu')
        return cached_latent


class MAVINDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            processor,
            prompt: str = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 16,
            fps: int = None,
            sample_frame_rate: int = None,
            use_random_frame_rate: bool = False,
            do_flip: bool = False,
            video_root: str = "training_videos",
            tokenizer_max_length: int = 77,
            fix_start_pos_rate: float = 0.0,
            condition_frame_index: int = 0,
            random_condition_frame: bool = False,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_root = video_root

        self.prompt = prompt

        if not isinstance(video_root, str):
            assert not isinstance(prompt, str)
            video_prompt_pairs = []
            video_files = {}
            for i, v in enumerate(video_root):
                video_files[prompt[i]] = glob(f"{v}/*.mp4")
                # flatten and append all to video_prompt_pairs
                video_prompt_pairs.extend([(x, prompt[i]) for x in video_files[prompt[i]]])
            self.video_prompt_pairs = video_prompt_pairs
        else:
            self.video_files = glob(f"{video_root}/*.mp4")
            if 'clip2' in self.video_files[0]:
                self.video_files[0] = self.video_files[0].replace('clip2', 'clip1')
                self.video_files[1] = self.video_files[1].replace('clip1', 'clip2')
            self.is_connection = 'clip' in self.video_files[0]

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps
        self.sample_frame_rate = sample_frame_rate
        self.use_random_frame_rate = use_random_frame_rate
        self.do_flip = do_flip
        self.tokenizer_max_length = tokenizer_max_length
        self.fix_start_pos_rate = fix_start_pos_rate
        self.condition_frame_index = condition_frame_index

        self.random_condition_frame = random_condition_frame

    def __len__(self):
        if not isinstance(self.video_root, str):
            return len(self.video_prompt_pairs)
        else:
            return len(self.video_files)

    def batch_by_sample_rate(self, vr, sample_frame_rate, random_frame_rate=False, return_all_possible=False):
        max_sample_rate = (len(vr) - 1) // (self.n_sample_frames - 1)
        sample_frame_rate = min(max_sample_rate, sample_frame_rate)
        if max_sample_rate == 0:
            # print(f"Video {vr} is too short to sample {self.n_sample_frames} frames at {sample_frame_rate} fps")
            return None
        if random_frame_rate:
            # sample rate is randomly selected within the given sample rate value
            sample_frame_rate = random.randint(1, max(1, sample_frame_rate))

        required_length = (self.n_sample_frames - 1) * sample_frame_rate + 1  # e.g. sample 3 frames at 4fps: xoooxooox = 2*4+1
        free_length = len(vr) - required_length

        if return_all_possible:
            all_possible_list = []
            for start_idx in range(free_length):
                sample_index = list(range(start_idx, len(vr), sample_frame_rate))[:self.n_sample_frames]
                video = vr.get_batch(sample_index)
                all_possible_list.append(video)
            return all_possible_list

        else:
            if random.uniform(0, 1) > self.fix_start_pos_rate:
                start_idx = random.randint(0, free_length)
            else:
                start_idx = 0
            sample_index = list(range(start_idx, len(vr), sample_frame_rate))[:self.n_sample_frames]
            video = vr.get_batch(sample_index)
            return video

    def batch_by_fps(self, vr, fps, return_all_possible=False):
        sample_frame_rate = max(1, round(vr.get_avg_fps() / fps))
        return self.batch_by_sample_rate(vr, sample_frame_rate, return_all_possible=return_all_possible)

    def get_frame_batch(self, vr, resize=None):
        if self.fps is not None:
            video = self.batch_by_fps(vr, self.fps)
        else:
            video = self.batch_by_sample_rate(vr, self.sample_frame_rate, self.use_random_frame_rate)

        if video is None:
            return video

        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None:
            video = resize(video)
        return video

    def process_video(self, vid_path):
        vr = decord.VideoReader(vid_path, width=self.width, height=self.height)
        video = self.get_frame_batch(vr)
        return video

    def get_prompt_ids(self, prompt):
        input_ids = self.tokenizer(prompt,
                                   truncation=True,
                                   padding=False,  # "max_length"
                                   # max_length=self.tokenizer_max_length,
                                   return_tensors="pt"
                                   ).input_ids[0]
        return input_ids

    def get_image_inputs(self, video):
        if self.random_condition_frame:
            condition_frame_index = random.randint(0, len(video) - 1)
        else:
            condition_frame_index = self.condition_frame_index

        inputs = self.processor(images=video[condition_frame_index],
                                return_tensors="pt"
                                ).pixel_values
        return inputs, condition_frame_index

    def __getitem__(self, index):
        if not isinstance(self.video_root, str):
            video_path, prompt = self.video_prompt_pairs[index]
        else:
            video_path = self.video_files[index]
            prompt = self.prompt
        video = self.process_video(video_path)

        if video is None:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        if self.do_flip and random.uniform(0, 1) > 0.5:
            video = torch.flip(video, dims=[3])

        prompt_ids = self.get_prompt_ids(prompt) if prompt else -999
        image, condition_frame_index = self.get_image_inputs(video)

        return {"video_values": video / 127.5 - 1.0, "prompt_ids": prompt_ids, "text_prompt": prompt,
                "image_values": image, "condition_frame_index": condition_frame_index}


class ConnectionDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            processor,
            prompt: str = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 16,
            fps: int = None,
            sample_frame_rate: int = None,
            do_flip: bool = False,
            video_root: str = "training_videos",
            prediction_start: int = None,
            prediction_end: int = None,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_root = video_root
        self.prompt = prompt
        self.prompt_cache = {}

        if not isinstance(video_root, str):
            # if passed as a list of strings
            self.video_files = []
            for v in video_root:
                self.video_files.extend(glob(f"{v}/*.mp4"))
        else:
            self.video_files = glob(f"{video_root}/*.mp4")

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps
        self.sample_frame_rate = sample_frame_rate
        self.do_flip = do_flip

        self.prediction_start = prediction_start
        self.prediction_end = prediction_end

        # Dump all valid training clips to a file
        self.metric = kwargs.get('metric', 'flow')
        self.motion_dir = kwargs.get('motion_dir', './temp_motion')
        motion_file_name = kwargs.get('motion_file_name', f'{self.metric}_{self.width}.json')
        self.motion_file = os.path.join(self.motion_dir, motion_file_name)
        if kwargs.get('make_dataset', False) or not os.path.exists(self.motion_file):
            assert not os.path.exists(self.motion_file), f"{self.motion_file} already exists"
            print(f"Preprocessing training data.")
            self.raft = torchvision.models.optical_flow.raft_large(pretrained=True).cuda()
            self.get_all_valid_clips(dynamic_fps=self.fps is not None, compute_motion=not kwargs.get('random_mask', False))

        with open(self.motion_file, 'r') as f:
            pre_clips = json.load(f)
            self.motion_selection = kwargs.get('motion_selection', None)
            if not kwargs.get('random_mask', False) and self.motion_selection:
                pre_clips = [x for x in pre_clips if sum(x[3]) / len(x[3]) > self.motion_selection]
                print(f"{len(pre_clips)} data points with motion intensity > {self.motion_selection}")
            self.pre_clips = pre_clips

    @staticmethod
    def calc_optical_flow(model, frames_t, distance, chunk_size=300):
        with torch.no_grad():
            with torch.backends.cudnn.flags(enabled=False):
                if len(frames_t[distance:]) > chunk_size:  # process chunk by chunk
                    all_motion_gaps = []
                    for j in range(0, len(frames_t[distance:]), chunk_size):
                        chunk_frames_t1 = frames_t[:-distance][j:j + chunk_size].contiguous().cuda()
                        chunk_frames_t2 = frames_t[distance:][j:j + chunk_size].contiguous().cuda()
                        all_motion_gaps.append(torch.norm(model(chunk_frames_t1, chunk_frames_t2)[-1], dim=1).mean(dim=(1, 2)))
                    all_motion_gaps = torch.cat(all_motion_gaps)
                else:
                    all_motion_gaps = torch.norm(model(frames_t[:-distance].contiguous().cuda(), frames_t[distance:].contiguous().cuda())[-1], dim=1).mean(dim=(1, 2))
        return all_motion_gaps

    def get_motion_diff(self, model, frames_t, distance, metric):
        if metric == 'ssim':
            motion_diff = 1 - ssim(frames_t[distance:], frames_t[:-distance], data_range=255, size_average=False)
        elif metric == 'ms_ssim':
            motion_diff = 1 - ms_ssim(frames_t[distance:], frames_t[:-distance], data_range=255, size_average=False)
        else:
            frames_t = frames_t / 127.5 - 1.0
            motion_diff = self.calc_optical_flow(model, frames_t, distance)

        return motion_diff

    def get_all_valid_clips(self, dynamic_fps=True, compute_motion=False):
        metric = self.metric
        model = self.raft
        model.eval()
        camera_switch_threshold = 0.1  # similarity smaller than this suggests a potential camera switch

        valid_clips = []
        for video_path in tqdm(self.video_files):

            vr = decord.VideoReader(video_path, width=self.width, height=self.height)
            total_frames = len(vr)

            sample_rate = max(1, round(vr.get_avg_fps() / self.fps)) if dynamic_fps else self.sample_frame_rate
            max_sample_rate = (len(vr) - 1) // (self.n_sample_frames - 1)
            sample_rate = min(max_sample_rate, sample_rate)
            if sample_rate == 0:  # video is too short to sample
                continue
            required_length = (self.n_sample_frames - 1) * sample_rate + 1  # e.g. sample 3 frames at 4fps: xoooxooox = 2*4+1

            frames = vr.get_batch(range(total_frames))
            frames_t = rearrange(frames, "f h w c -> f c h w").float()

            # Calculate for all frames at once
            all_scene_sim = ssim(frames_t[1:], frames_t[:-1], data_range=255, size_average=False)
            if compute_motion:
                distance = (self.prediction_end - self.prediction_start + 1) * sample_rate
                all_motion_gaps = self.get_motion_diff(model, frames_t, distance, metric)
                # all_motion_gaps = self.get_motion_diff(model, frames_t, sample_rate, metric)

            for start_frame in range(0, total_frames - required_length + 1):  # this ensures not to go over the last frame
                # Check if the clip contains a potential camera switch
                if (all_scene_sim[start_frame: start_frame + required_length - 1] < camera_switch_threshold).any():
                    print(video_path, start_frame)
                    continue

                if compute_motion:
                    motion_intensity = all_motion_gaps[start_frame + (self.prediction_start - 1) * sample_rate]
                    # all_clip_motions = all_motion_gaps[start_frame: start_frame + required_length - 1: sample_rate]
                    valid_clips.append((str(video_path), int(start_frame), int(sample_rate), motion_intensity))
                else:
                    valid_clips.append((str(video_path), int(start_frame), int(sample_rate), None))

            os.makedirs(self.motion_dir, exist_ok=True)
            with open(self.motion_file, 'w') as fw:
                json.dump(valid_clips, fw)
            torch.cuda.empty_cache()

    def get_prompt_ids(self, prompt):
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt]

        input_ids = self.tokenizer(prompt,
                                   truncation=True,
                                   # padding=False,
                                   padding="max_length",
                                   max_length=self.tokenizer.model_max_length,
                                   return_tensors="pt"
                                   ).input_ids[0]

        self.prompt_cache[prompt] = input_ids
        return input_ids

    def __len__(self):
        return len(self.pre_clips)

    def __getitem__(self, index):
        tp = self.pre_clips[index]
        video_path, start_frame, sr, _ = tp

        vr = decord.VideoReader(video_path, width=self.width, height=self.height)
        sr = self.sample_frame_rate if sr > self.sample_frame_rate else sr
        frames = [start_frame + i * sr for i in range(self.n_sample_frames)]
        video = vr.get_batch(frames)
        video = rearrange(video, "f h w c -> f c h w")
        prompt = self.prompt

        if self.do_flip and random.uniform(0, 1) > 0.5:
            video = torch.flip(video, dims=[3])

        prompt_ids = self.get_prompt_ids(prompt)

        return {"video_values": video / 127.5 - 1.0, "prompt_ids": prompt_ids}
