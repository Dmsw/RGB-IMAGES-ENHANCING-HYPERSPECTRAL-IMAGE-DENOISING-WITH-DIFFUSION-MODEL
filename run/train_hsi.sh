CUDA_VISIBLE_DEVICES=0 python ../scripts/hsi_train.py \
    --data_dir "../../dataset/cave/randga75/" \
    --batch_size 4 \
    --save_interval 10000 \
    --model_config ../config/model_config.yaml \
    --lr_anneal_steps 30000 \
    --save_dir ../models/hsi256_new2/
