outputdir=saved_models/

echo $outputdir
mkdir -p $outputdir

accelerate launch --mixed_precision=fp16 \
    --use_deepspeed \
    --zero_stage 2 --offload_param_device none --offload_optimizer_device none --gradient_accumulation_steps 1 --zero3_init_flag false \
    train.py \
    --pretrained_model_name_or_path=weights/stable-diffusion-v1-5 \
    --unet weights/sd_13channel_stage1_pretrain \
    --output_dir $outputdir \
    --metafiles '' \
    --img_metafiles '' \
    --mask_drop 0.25 \
    --image_skip_motion True \
    --all_learn 1 \
    --use_grad_loss 1 \
    --num_frames 8 \
    --num_repeats 2 \
    --clip_skip 2 \
    --save_steps 10000 \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=1 \
    --dataloader_num_workers=6 \
    --num_train_epochs=1000 \
    --mixed_precision=fp16 \
    --seed 11145

