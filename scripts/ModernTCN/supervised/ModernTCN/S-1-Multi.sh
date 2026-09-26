#!/usr/bin/env bash


# Multi-class classification
# ADFTD
python -u run.py \
  --method ModernTCN \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFTD-Multi \
  --model ModernTCN \
  --data MultiDatasets \
  --training_dataset ADFTD \
  --sampling_rate_list 200 \
  --ffn_ratio 1 \
  --patch_len 32 \
  --stride 16 \
  --num_blocks 1 1 1 1 \
  --large_size 9 9 9 9 \
  --small_size 5 5 5 5 \
  --dims 32 64 128 128 \
  --batch_size 512 \
  --classify_choice multi_class \
  --use_subject_vote \
  --swa \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# CNBPM
python -u run.py \
  --method ModernTCN \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-CNBPM-Multi \
  --model ModernTCN \
  --data MultiDatasets \
  --training_dataset CNBPM \
  --sampling_rate_list 200 \
  --ffn_ratio 1 \
  --patch_len 32 \
  --stride 16 \
  --num_blocks 1 1 1 1 \
  --large_size 9 9 9 9 \
  --small_size 5 5 5 5 \
  --dims 32 64 128 128 \
  --batch_size 512 \
  --classify_choice multi_class \
  --use_subject_vote \
  --swa \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

## APAVA
python -u run.py \
  --method ModernTCN \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-APAVA-Multi \
  --model ModernTCN \
  --data MultiDatasets \
  --training_dataset APAVA \
  --sampling_rate_list 200 \
  --ffn_ratio 1 \
  --patch_len 32 \
  --stride 16 \
  --num_blocks 1 1 1 1 \
  --large_size 9 9 9 9 \
  --small_size 5 5 5 5 \
  --dims 32 64 128 128 \
  --batch_size 128 \
  --classify_choice multi_class \
  --use_subject_vote \
  --swa \
  --cross_val fixed \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# ADFSU
python -u run.py \
  --method ModernTCN \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFSU-Multi \
  --model ModernTCN \
  --data MultiDatasets \
  --training_dataset ADFSU \
  --ffn_ratio 1 \
  --patch_len 20 \
  --stride 10 \
  --num_blocks 1 1 1 1 \
  --large_size 9 9 9 9 \
  --small_size 5 5 5 5 \
  --dims 32 64 128 128 \
  --batch_size 128 \
  --classify_choice multi_class \
  --use_subject_vote \
  --swa \
  --sampling_rate_list 100 \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# ADSZ
python -u run.py \
  --method ModernTCN \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADSZ-Multi \
  --model ModernTCN \
  --data MultiDatasets \
  --training_dataset ADSZ \
  --ffn_ratio 1 \
  --patch_len 20 \
  --stride 10 \
  --num_blocks 1 1 1 1 \
  --large_size 9 9 9 9 \
  --small_size 5 5 5 5 \
  --dims 32 64 128 128 \
  --batch_size 128 \
  --classify_choice multi_class \
  --use_subject_vote \
  --swa \
  --sampling_rate_list 100 \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15
