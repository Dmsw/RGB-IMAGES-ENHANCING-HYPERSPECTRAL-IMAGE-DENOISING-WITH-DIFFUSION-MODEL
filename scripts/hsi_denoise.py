import argparse
import os

import blobfile as bf
import scipy.io as scio
import numpy as np
import torch as th
import torch.distributed as dist
import yaml
import cv2 as cv

import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    )
from guided_diffusion.image_datasets import load_hsi_data
from torchvision import utils
from measurement import measurement_fn
from utils import calc_psnr, calc_ssim, calc_sam


def estimate_sigma(image, k, k_m=1):
    assert k > k_m
    image = np.array(image.to("cpu"))
    image = np.transpose(image, [1, 2, 0])
    blur = cv.blur(image, (k, k))
    v = np.var(blur - image, axis=(0, 1))
    std = np.sqrt(v*(k**2)/(k**2-k_m**2))
    return th.from_numpy(std).to(dist_util.dev())


def estimate_cov(image, k, k_m=1):
    assert k > k_m
    image = np.array(image.to("cpu"))
    image = np.transpose(image, [1, 2, 0])
    blur = cv.blur(image, (k, k))
    noise = blur - image
    noise = noise.reshape([-1, noise.shape[-1]])
    cov = noise.T @ noise / noise.shape[0] * (k**2) / (k**2 - k_m**2)
    cov[np.abs(cov) < 2e-3] = 0
    return th.from_numpy(cov.astype(np.float32)).to(dist_util.dev())


def calc_covariance(noise, true):
    noise = noise.reshape([noise.shape[0], -1])
    true = true.reshape([true.shape[0], -1])
    noise = noise - true
    cov = noise @ noise.T / noise.shape[1]
    return cov

# added
def load_noise_hsi(data_dir, batch_size):
    data = load_hsi_data(
        data_dir=data_dir,
        batch_size=batch_size,
        deterministic=True,
        mode="test",
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch[1][0]
        yield model_kwargs, large_batch[0][0]


def load_P():
    P = scio.loadmat("/home/root/dataset/cave/P.mat")['P']
    return th.from_numpy(P.astype(np.float32)).to(dist_util.dev())


def main():
    args = create_argparser().parse_args()
    if args.model_config is not None:
        upgrade_by_config(args, args.model_config)
    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    logger.log(args)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    """
    load rgb model
    """
    if args.rgb_model_config is not None:
        logger.log("creating rgb model...")
        upgrade_by_config(args, args.rgb_model_config)
        rgb_model, rgb_diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
        rgb_model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        rgb_model.to(dist_util.dev())
        if args.use_fp16:
            rgb_model.convert_to_fp16()
        rgb_model.eval()
    else:
        rgb_model = None

    logger.log("loading data...")
    data = load_noise_hsi(
        args.base_samples,
        args.batch_size,
    )

    """
    load P matrix
    """
    P = load_P()

    logger.log("creating samples...")
    count = 0
    apsnr = 0
    assim = 0
    asam = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs, target = next(data)
        cov = calc_covariance(model_kwargs["ref_img"], target)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        cov = cov.to(dist_util.dev())
        # sigma = estimate_sigma(model_kwargs["ref_img"], args.k)
        # print("err:", th.mean((sigma**2 - th.diagonal(cov))**2))
        sample, distance, rgb_distance, rgb_t, hsi_t = diffusion.p_sample_loop(
            model=model,
            rgb_model=rgb_model,
            P=P,
            l1=args.l1,
            l2=args.l2,
            rgb_diffusion=rgb_diffusion,
            shape=(args.batch_size, 31, *target.shape[-2:]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            measure_fn=measurement_fn,
            range_t=args.range_t,
            progress=True,
            var=cov,
            # var = th.diag(sigma**2).to(dist_util.dev()),      
            # var = estimate_cov(model_kwargs["ref_img"], args.k),  
        )
        
        # plt.figure()
        # plt.plot(distance[::-1], label="hsi")
        # plt.plot(rgb_distance[::-1], label="rgb")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(logger.get_dir(), f"{str(count).zfill(5)}_distance.png"))
        # plt.close()

        # plt.figure()
        # plt.plot(diff[::-1], label="diff")
        # plt.savefig(os.path.join(logger.get_dir(), f"{str(count).zfill(5)}_diff.png"))
        np.savez(os.path.join(logger.get_dir(), f"{str(count).zfill(5)}_distance.npz"), hsi=distance, rgb=rgb_distance)
        # plt.close()

        sample = (sample + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"{str(count).zfill(5)}.png")
        utils.save_image(
            sample[0, 0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )

        out_path = os.path.join(logger.get_dir(),
                                f"{str(count).zfill(5)}.npy")
        np.save(out_path, sample.cpu().numpy())

        target = (target + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"target-{str(count).zfill(5)}.png")
        utils.save_image(
            target[0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )

        noise = model_kwargs['ref_img']
        noise = (noise + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"noise-{str(count).zfill(5)}.png")
        utils.save_image(
            noise[0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )
        target = target.numpy()
        noise = noise.cpu().numpy()
        sample = sample.cpu().numpy()[0]
        apsnr += calc_psnr(target, sample)
        assim += calc_ssim(target, sample)
        asam += calc_sam(target, sample)
        logger.log("PSNR:")
        logger.log(f"count:{count}, before: {calc_psnr(target, noise)}")
        logger.log(f"count:{count}, after: {calc_psnr(target, sample)}")
        logger.log("SSIM:")
        logger.log(f"count:{count}, before: {calc_ssim(target, noise)}")
        logger.log(f"count:{count}, after: {calc_ssim(target, sample)}")
        logger.log("SAM:")
        logger.log(f"count:{count}, before: {calc_sam(target, noise)}")
        logger.log(f"count:{count}, after: {calc_sam(target, sample)}")

        logger.log(f"created {count * args.batch_size} samples")

        # t_psnr = []
        # t_ssim = []
        # for hsi in hsi_t:
        #     hsi = (hsi+1)/2
        #     t_psnr.append(calc_psnr(target, hsi))
        #     t_ssim.append(calc_ssim(target, hsi))
        # t_psnr = np.array(t_psnr)
        # t_ssim = np.array(t_ssim)

        # np.savez(os.path.join(logger.get_dir(), f"{str(count).zfill(5)}_theory.npz"), psnr=t_psnr, ssim=t_ssim, hsi=hsi_t, rgb=rgb_t)

        count += 1

    dist.barrier()
    logger.log("sampling complete")
    apsnr /= count
    assim /= count
    asam /= count
    logger.log(f"average PSNR: {apsnr}")
    logger.log(f"average SSIM: {assim}")
    logger.log(f"average SAM: {asam}")



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5,
        batch_size=1,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        model_config=None,
        rgb_model_config=None,
        k=5,
        l1=2.0,
        l2=0.1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def upgrade_by_config(args, model_config):
    model_config = load_yaml(model_config)
    for k, v in model_config.items():
        setattr(args, k, v)


if __name__ == "__main__":
    main()