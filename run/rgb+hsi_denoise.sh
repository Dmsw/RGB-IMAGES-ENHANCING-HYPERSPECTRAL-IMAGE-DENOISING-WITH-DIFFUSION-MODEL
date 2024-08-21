noise_model="randcov75"
save_name="RGB+HSI_DM4"
dataset="cave"
cuda=1

# base_samples: path to dataset
# save_dir: path to save the result
# model_config: path to HSI DM config
# in_channels: number of bands
# range_t: perform unconditional denoising in the last range_t steps
# num_samples: set to the size of dataset
# rgb_model_config: path to RGB DM config
# l1: index for gradient
# l2: fusion weight of RGB DM

CUDA_VISIBLE_DEVICES=$cuda python ../scripts/hsi_denoise.py \
 --base_samples /home/root/dataset/$dataset/$noise_model/ \
 --save_dir ../results/new_hsi/$dataset/$noise_model/$save_name/ \
 --model_config ../config/model_config.yaml \
 --in_channels 31 \
 --range_t 0 \
 --num_samples 3 \
 --rgb_model_config ../config/model_config_rgb.yaml \
 --l1 2 \
 --l2 0.52
