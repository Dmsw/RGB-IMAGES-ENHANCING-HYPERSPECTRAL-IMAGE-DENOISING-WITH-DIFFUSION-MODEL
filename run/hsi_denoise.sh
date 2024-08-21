CUDA_VISIBLE_DEVICES=1 python ../scripts/hsi_denoise.py \
 --base_samples "/home/root/dataset/cave/randga75/" \
 --save_dir "../results/cave/randga75/wo_rgb/" \
 --model_config ../config/model_config.yaml \
 --in_channels 31 \
 --range_t 0 \
 --num_samples 3 \
 --l1 2 
