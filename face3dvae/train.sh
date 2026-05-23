accelerate launch --mixed_precision=fp16 \
    train.py \
    --pretrained_model_name_or_path=/path/to/sd_pretrain_weights \
    --dataloader_num_workers 8\
    --train_batch_size 2 \
    --validation_steps 2000 \
    --checkpointing_steps 10000 \
    --output_dir /path/to/save \
    --learning_rate 5e-6 \
    --use_ema
