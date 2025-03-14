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

To run this project,

1. install requirements:
```
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
$ pip install PyYAML opencv-python scipy blobfile matplotlib h5py tqdm
$ conda install mpi4py
```

2. train the HSI DM using the scripts\hsi_train.py. In run\train_hsi.sh, you can find an example for training. To train on your dataset, please modify the class HSIDataset and function load_hsi_data in guided_diffusion\image_datasets.py.
```
$ cd run
$ bash train.sh
```

3. download the RGB DM (https://github.com/openai/guided-diffusion.git) and the corresponding configuration is set in config\model_config_rgb.yaml. Please change the model_path in this file to the path to the downloaded RGB DM.

4. run scripts\hsi_denoise.py for denoising. In run\rgb+hsi_denoise.sh, you can find an exmaple for HSI denoising with RGB DM enhanced. 
```
$ cd run
$ bash rgb+hsi_denoise.sh
```

