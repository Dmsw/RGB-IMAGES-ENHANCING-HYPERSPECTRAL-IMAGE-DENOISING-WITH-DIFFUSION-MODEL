# RGB-IMAGES-ENHANCING-HYPERSPECTRAL-IMAGE-DENOISING-WITH-DIFFUSION-MODEL
This project is highly dependent on guided-diffusion-model (https://github.com/openai/guided-diffusion.git).
The RGB DM used in the paper is the 256x256_diffusion_uncond.pt (https://github.com/openai/guided-diffusion.git).

some important files and directories:
```
config\                       # configuration for RGB DM and HSI DM
guided-diffusion\
  gaussian_diffusion.py       # diffusion model for denoising and loss function
  image_datasets.py           # dataloader
  script_util.py              # some default config
  train_util.py               # training function for diffusion model
run\
  hsi_denoise.sh              # HSI denoising without RGB DM
  rgb+hsi_denoise.sh          # RGB+HSI denoising
  train_hsi.sh                # HSI DM training
scripts\
  generate_test_data.py       # generate the test data
  hsi_denoise.py              # HSI denoising
  hsi_train.py                # train the HSI DM
  image_train.py              # train the RGB DM
  
measurement.py                # gradient of log-posterior
utils.py                      # some auxiliary functions
```
  
