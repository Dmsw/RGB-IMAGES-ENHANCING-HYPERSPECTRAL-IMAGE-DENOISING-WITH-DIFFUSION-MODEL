import torch
from guided_diffusion.dist_util import dev
import numpy as np
import cv2 as cv
from utils import tensor_2_mode_product


def measurement_fn(x_t, ref, bar_alpha_t, var, alpha_t):
    device = x_t.device
    over_sqrt_bar_alpha_t = (1 / np.sqrt(bar_alpha_t))
    coeff = bar_alpha_t * torch.inverse(var * bar_alpha_t + (1 - bar_alpha_t) * torch.eye(var.shape[0], device=device))
    return over_sqrt_bar_alpha_t * tensor_2_mode_product(ref - over_sqrt_bar_alpha_t * x_t, coeff)


