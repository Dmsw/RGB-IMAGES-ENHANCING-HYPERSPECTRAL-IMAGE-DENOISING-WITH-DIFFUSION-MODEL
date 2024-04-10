import numpy as np
import scipy.io as sio
import os
from os.path import join
import h5py
import tqdm
import sys
sys.path.append("../")
from utils import calc_psnr, calc_ssim

data = np.load("../../dataset/icvl/randga75/test.npz")
P = sio.loadmat("/home/root/dataset/cave/P.mat")['P']

clean_hsi = data['clean_img']
noise_hsi = data['noise_img']

n, w, h, c= clean_hsi.shape

noise_hsi = clean_hsi.copy()
covs = []
for i in range(n):
    p = np.random.uniform(0, 1, [c, c]) * 75 / 255
    cov = p @ p.T / c
    noise = np.random.multivariate_normal(np.zeros(c), cov, size=(w, h))
    noise_hsi[i] += noise

save_dir = "../../dataset/icvl/randcov75"
if os.path.exists(save_dir) == False:
    os.mkdir(save_dir)
np.savez(os.path.join(save_dir, "test.npz"), clean_img=clean_hsi, noise_img=noise_hsi, cov=covs)
