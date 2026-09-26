#!/usr/bin/env bash

# Finetuning


# Multi-Class Classification
# ADFTD
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-fixed/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-g16-learnable-fixed-F-ADFTD-Multi --model LEAD --data MultiDatasets \
--training_dataset ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type fixed --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 4 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-fixed/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-g16-learnable-fixed-F-CNBPM-Multi --model LEAD --data MultiDatasets \
--training_dataset CNBPM \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type fixed --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# APAVA
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-fixed/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-g16-learnable-fixed-F-APAVA-Multi --model LEAD --data MultiDatasets \
--training_dataset APAVA \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type fixed --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADSZ
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-fixed/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-g16-learnable-fixed-F-ADSZ-Multi --model LEAD --data MultiDatasets \
--training_dataset ADSZ \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type fixed --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADFSU
python -u run.py --method LEAD --checkpoints_path ./checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-fixed/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-g16-learnable-fixed-F-ADFSU-Multi --model LEAD --data MultiDatasets \
--training_dataset ADFSU \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type fixed --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
