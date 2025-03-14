CUDA_VISIBLE_DEVICES=0 python ../scripts/hsi_train.py \
    --data_dir /PATH/TO/DATASET/ \
    --batch_size 4 \
    --save_interval 10000 \
    --model_config ../config/model_config.yaml \
    --lr_anneal_steps 30000 \
    --save_dir /PATH/TO/SAVE/MODEL/
