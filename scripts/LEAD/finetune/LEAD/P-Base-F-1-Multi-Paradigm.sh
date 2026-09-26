#!/usr/bin/env bash

# Finetuning


# Multi-Class Classification
# ADFTD-RS
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-3D/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-g16-learnable-3D-F-ADFTD-RS-Multi --model LEAD --data MultiDatasets \
--training_dataset ADFTD-RS \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 4 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ADFTD-PS
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-3D/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-g16-learnable-3D-F-ADFTD-PS-Multi --model LEAD --data MultiDatasets \
--training_dataset ADFTD-PS \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
