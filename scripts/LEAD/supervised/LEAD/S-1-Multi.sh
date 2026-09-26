#!/usr/bin/env bash

# Finetuning


# Multi-Class Classification
# ADFTD
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-ADFTD-Multi --model LEAD --data MultiDatasets \
--training_dataset ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 4 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-CNBPM-Multi --model LEAD --data MultiDatasets \
--training_dataset CNBPM \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# APAVA
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-APAVA-Multi --model LEAD --data MultiDatasets \
--training_dataset APAVA \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADSZ
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-ADSZ-Multi --model LEAD --data MultiDatasets \
--training_dataset ADSZ \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADFSU
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-ADFSU-Multi --model LEAD --data MultiDatasets \
--training_dataset ADFSU \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --group_shuffle --group_size 2 --use_subject_loss \
--use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa --classify_choice multi_class \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
