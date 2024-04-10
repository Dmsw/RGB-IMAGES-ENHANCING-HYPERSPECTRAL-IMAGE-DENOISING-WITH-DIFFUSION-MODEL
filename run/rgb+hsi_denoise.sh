noise_model="randcov75"
save_name="RGB+HSI_DM4"
dataset="cave"
cuda=1

CUDA_VISIBLE_DEVICES=$cuda python ../scripts/hsi_denoise.py \
 --base_samples /home/root/dataset/$dataset/$noise_model/ \
 --save_dir ../results/new_hsi/$dataset/$noise_model/$save_name/ \
 --model_config ../models/hsi256/model_config.yaml \
 --in_channels 31 \
 --range_t 0 \
 --num_samples 3 \
 --rgb_model_config ../models/DM256/model_config.yaml \
 --l1 2 \
 --l2 0.52