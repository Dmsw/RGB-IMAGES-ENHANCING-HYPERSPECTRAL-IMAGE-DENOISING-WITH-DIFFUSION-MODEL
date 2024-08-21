noise_model="randcov75"
save_name="RGB+HSI_DM4"
dataset="cave"
cuda=1

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
