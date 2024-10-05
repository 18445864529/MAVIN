from glob import glob
from tqdm import tqdm
import os
import torch
from transformers import CLIPProcessor, CLIPVisionModel
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from mavin.utils.common_metrics_on_video_quality.calculate_fvd import calculate_fvd



def metric_calculation(exp_dir, gt_dir, n=12, step=30000, resolution=256, baseline=False, processor=None, encoder=None, write=False, dyna='1'):
    if n == 8:
        start, end = 12, 20
    elif n == 12:
        start, end = 10, 22
    elif n == 6:
        start, end = 13, 19
    else:
        raise ValueError("n must be 8 or 12")

    import decord
    decord.bridge.set_bridge('torch')
    print(f"======================Evaluating step {step}======================")

    ssim = SSIM(data_range=255)
    ms_ssim = MS_SSIM(data_range=255)
    lpips = LPIPS(net_type='squeeze')
    psnr = PSNR()

    if baseline:
        pred_files = sorted(glob(os.path.join(exp_dir, f"*.mp4")))
    else:
        pred_files = sorted(glob(os.path.join(exp_dir, f"*{step}*.mp4")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*.mp4")))

    results = {}
    pairs = {}

    for gf in gt_files:
        if '_' in gf:
            base_name_mp4 = os.path.basename(gf).split("_")[-1]
        else:
            base_name_mp4 = os.path.basename(gf)
        base_name = base_name_mp4.split(".")[0]
        if 'clip' in base_name_mp4:
            continue
        for pf in pred_files:
            # if baseline == 'seine':
            #     p_dir = pf.split("/")[-2]
            #     if p_dir == base_name:
            #         pairs[base_name_mp4] = (pf, gf)
            if baseline:
                if base_name_mp4 == pf.split('/')[-1]:
                    pairs[base_name_mp4] = (pf, gf)
            else:
                base_name2 = os.path.basename(pf)
                if base_name2.endswith(f"_{base_name_mp4}"):
                    pairs[base_name_mp4] = (pf, gf)

    pred_samples = []
    gt_samples = []

    for base_name, pair in pairs.items():

        vr_pred = decord.VideoReader(pair[0], width=resolution, height=resolution)
        vr_gt = decord.VideoReader(pair[1], width=resolution, height=resolution)

        if baseline == 'seine':
            if len(vr_pred) == 32:
                pred = vr_pred.get_batch(range(start-1, end+1)).permute(0, 3, 1, 2).float()
            else:
                pred = vr_pred.get_batch(range(0, len(vr_pred))).permute(0, 3, 1, 2).float()
        elif baseline == 'dynami':
            if dyna == '2':
                if end - start == 8:
                    pred = vr_pred.get_batch(range(3, 13)).permute(0, 3, 1, 2).float()
                elif end - start == 6:
                    pred = vr_pred.get_batch(range(4, 12)).permute(0, 3, 1, 2).float()
                else:
                    assert end - start == 12
                    pred = vr_pred.get_batch(range(1, 15)).permute(0, 3, 1, 2).float()
            else:
                all_frames_idx = torch.arange(1, len(vr_pred) - 1)
                indices = torch.linspace(0, len(all_frames_idx) - 1, end - start).long()
                sampled_frames = torch.index_select(all_frames_idx, 0, indices)
                sf_boundry = [0] + sampled_frames + [len(vr_pred) - 1]
                pred = vr_pred.get_batch(sf_boundry).permute(0, 3, 1, 2).float()
        else:
            pred = vr_pred.get_batch(range(start-1, end+1)).permute(0, 3, 1, 2).float()  # rearrange(video, "f h w c -> f c h w")
        gt = vr_gt.get_batch(range(start-1, end+1)).permute(0, 3, 1, 2).float()

        # pred_samples.append(pred.permute(1, 0, 2, 3))
        # gt_samples.append(gt.permute(1, 0, 2, 3))
        pred_samples.append(vr_pred.get_batch(range(len(vr_pred))).float().permute(3, 0, 1, 2))
        gt_samples.append(vr_gt.get_batch(range(len(vr_gt))).float().permute(3, 0, 1, 2))

        pred_mid = pred[1:-1]
        gt_mid = gt[1:-1]
        s = ssim(pred_mid, gt_mid)
        ms = ms_ssim(pred_mid, gt_mid)
        lp = lpips(pred_mid / 127.5 - 1, gt_mid / 127.5 - 1)
        ps = psnr(pred_mid, gt_mid)

        results[base_name] = {
            "ssim": s.item(),
            "ms_ssim": ms.item(),
            "psnr": ps.item(),
            "lpips": lp.item(),
        }

        if processor is not None and encoder is not None:
            with torch.no_grad():
                images_pred = processor(images=pred, return_tensors="pt").pixel_values.to("cuda")
                clip_states_pred = encoder(images_pred)[0]
                cossim_pred = torch.nn.functional.cosine_similarity(clip_states_pred[:-1], clip_states_pred[1:])
                images_gt = processor(images=gt, return_tensors="pt").pixel_values.to("cuda")
                clip_states_gt = encoder(images_gt)[0]
                cossim_gt = torch.nn.functional.cosine_similarity(clip_states_gt[:-1], clip_states_gt[1:])
                referred_cs = (torch.min(cossim_pred, cossim_gt) / torch.max(cossim_pred, cossim_gt)).mean()
                clip_inner = referred_cs
                clip_outer = torch.nn.functional.cosine_similarity(clip_states_pred, clip_states_gt).mean()
            results[base_name]["clip_inner"] = clip_inner.item()
            results[base_name]["clip_outer"] = clip_outer.item()

    a = torch.stack(gt_samples).permute(0, 2, 1, 3, 4).to(dtype=torch.float32)
    b = torch.stack(pred_samples).permute(0, 2, 1, 3, 4).to(dtype=torch.float32)
    if a.shape[1] > b.shape[1]:
        la = a.shape[1]
        lb = b.shape[1]
        diff = (la - lb) // 2
        a = a[:, diff:la - diff]
    fvd = calculate_fvd(a, b, torch.device('cuda'), method='videogpt')

    avg_ssim = sum([scores['ssim'] for scores in results.values()]) / len(results)
    avg_ms_ssim = sum([scores['ms_ssim'] for scores in results.values()]) / len(results)
    avg_lpips = sum([scores['lpips'] for scores in results.values()]) / len(results)
    avg_psnr = sum([scores['psnr'] for scores in results.values()]) / len(results)
    avg_fvd = fvd
    avg_cos_in = sum([scores['clip_inner'] for scores in results.values()]) / len(results) if processor is not None and encoder is not None else 0
    avg_cos_out = sum([scores['clip_outer'] for scores in results.values()]) / len(results) if processor is not None and encoder is not None else 0
    print(f"====================SSIM = {avg_ssim}, MS-SSIM = {avg_ms_ssim}, LPIPS = {avg_lpips}, PSNR = {avg_psnr}, FVD = {avg_fvd}====================")
    print(f"==================Average inner CLIP cosine similarity: {avg_cos_in}, Average outer CLIP cosine similarity: {avg_cos_out}==================\n")
    print(len(results))
    if write:
        with open(f"{exp_dir}/eval_log.txt", write) as f:
            f.write(f"Step {step}:\n")
            f.write(f"====================SSIM = {avg_ssim}, MS-SSIM = {avg_ms_ssim}, LPIPS = {avg_lpips}, PSNR = {avg_psnr}, FVD = {avg_fvd}====================\n")
            f.write(f"==================Average inner CLIP cosine similarity: {avg_cos_in}, Average outer CLIP cosine similarity: {avg_cos_out}==================\n")


if __name__ == '__main__':
    clip_image_repo = "openai/clip-vit-large-patch14"
    # processor = CLIPProcessor.from_pretrained(clip_image_repo)
    # vision_encoder = CLIPVisionModel.from_pretrained(clip_image_repo)
    # vision_encoder = vision_encoder.to("cuda")
    # vision_encoder.eval()
    # metric_calculation(bs_dir, gt_dir, baseline='seine', processor=processor, encoder=vision_encoder, write='w', n=8)

