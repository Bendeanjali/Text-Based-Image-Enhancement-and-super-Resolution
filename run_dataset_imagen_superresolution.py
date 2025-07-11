import argparse
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import get_Afuncs
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import pandas as pd
import numpy as np

# Set environment variables for GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Text-guided super-resolution")
    parser.add_argument("--log_dir", type=str, help="Directory for logs")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor")
    parser.add_argument("--count", type=int, default=100, help="Number of images to process")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--dec_steps", type=int, default=100, help="Decoder steps")
    parser.add_argument("--sr_steps", type=int, default=50, help="Super-resolution steps")
    parser.add_argument("-g1", "--guidance_scale_stage1", type=float, default=7.5, help="Guidance scale for stage 1")
    parser.add_argument("-g2", "--guidance_scale_stage2", type=float, default=4.0, help="Guidance scale for stage 2")
    parser.add_argument("--algo", type=str, choices=["ddnm", "dps", "pigdm"], help="Algorithm to use")
# Add missing arguments below:
    parser.add_argument("--dps_scale", type=float, default=0.5, help="DPS scale factor (specific to DPS algorithm)")
    parser.add_argument("--start_time", type=int, default=0, help="Start time for inference steps")

    return parser.parse_args()


def main(args):
    """Main function for text-guided super-resolution."""
    # Load models
    token = "your_huggingface_token_here"

    from pipeline_if import IFPipeline
    from pipeline_if_superresolution import IFSuperResolutionPipeline

    stage_1 = IFPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        variant="fp16",
        torch_dtype=torch.float16,
        use_auth_token=token
    )
    stage_1.enable_model_cpu_offload()

    stage_2 = IFSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
        use_auth_token=token
    )
    stage_2.enable_model_cpu_offload()

    # Load dataset
    obj = pd.read_pickle('./data/work_data/multi_mod_celebahq/filenames.pickle')
    image_dir = './data/work_data/multi_mod_celebahq/CelebAMask-HQ/CelebA-HQ-img/'

    # Image resizing (ground truth size: 256x256)
    H, W = 256, 256
    improcess = transforms.Compose([transforms.Resize((H, W), interpolation=InterpolationMode.BICUBIC)])

    # Consistency enforcement
    scale = args.scale
    Av = get_Afuncs('sr_bicubic', sizes=256, sr_scale=scale)
    A, Ap = Av.A, Av.A_pinv

    # Create output directory
    exp_dir = f'{args.log_dir}_{args.algo}_{args.scale}/'
    os.makedirs(exp_dir, exist_ok=True)

    # Quantitative evaluation
    for run in range(args.runs):
        print(f"Starting run {run + 1}/{args.runs}")
        exp_dir_run = f'{exp_dir}/run_{run}'
        os.makedirs(exp_dir_run, exist_ok=True)

        LR_psnr = []

        for i in range(args.count):
            file_num = obj[i]
            image_name = os.path.join(image_dir, file_num)

            # Load and preprocess ground truth image
            gt_t = Image.open(image_name).convert('RGB')
            gt_t = improcess(gt_t)
            gt_t = torch.tensor(np.array(gt_t).transpose(2, 0, 1)).float() / 255.0

            prompt = 'A professional realistic high-res portrait face photograph'
            negative_prompt = 'disfigured, blurred, ugly, bad, unnatural'

            # Encode prompts
            prompt_embeds, negative_embeds = stage_1.encode_prompt(
                prompt=prompt, negative_prompt=negative_prompt
            )

            # Bicubic downsampling
            lrf = A(gt_t.unsqueeze(0).cuda()).reshape(1, 3, H // scale, W // scale).half()

            # Stage 1: Text-guided restoration
            image = stage_1.sup_res(
                lr=lrf,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                num_inference_steps=args.dec_steps,
                guidance_scale=args.guidance_scale_stage1,
                algo=args.algo,
                output_type="pt",
                sr_scale=scale // 4
            )

            # Stage 2: Super-resolution
            image = stage_2.sup_res(
                image=image,
                lr=lrf,
                sr_scale=scale // 4,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                num_inference_steps=args.sr_steps,
                guidance_scale=args.guidance_scale_stage2,
                algo=args.algo,
                output_type="pt"
            ).images.clamp_(-1, 1)

            # Quantitative evaluation
            lroutfin = A(image.reshape(image.size(0), -1).float()).reshape(1, 3, H // scale, W // scale)
            LR_psnr.append(-10 * torch.log10(F.mse_loss(lroutfin, lrf)))

            # Save results
            res = (image[0].float().cpu()).permute(1, 2, 0).clip(0, 1.0).numpy()
            plt.imsave(f'{exp_dir_run}/{i}_result.jpg', res)

            lrsave = F.interpolate(
                lrf.float(), size=(H, W), mode='nearest'
            )[0].cpu().permute(1, 2, 0).clip(0, 1.0).numpy()
            plt.imsave(f'{exp_dir_run}/{i}_LR.jpg', lrsave)

        # Save evaluation metrics
        np.savetxt(f'{exp_dir_run}/LR_Psnr.txt', torch.Tensor(LR_psnr).numpy())
        stats = f'Mean LR_PSNR: {torch.Tensor(LR_psnr).mean().item()}'
        with open(f'{exp_dir_run}/stats.txt', "w") as text_file:
            text_file.write(stats)


if __name__ == '__main__':
    args = parse_args()
    main(args)
