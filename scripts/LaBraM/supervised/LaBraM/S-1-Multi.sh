#!/usr/bin/env bash

# Multi-Class Classification
# ADFTD
python -u run.py \
  --method LaBraM \
  --checkpoints_path ./checkpoints/LaBraM/pretrain_labram/LaBraM/labram-base.pth \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFTD-Multi \
  --model LaBraM \
  --data MultiDatasets \
  --training_dataset ADFTD \
  --sampling_rate_list 200 \
  --batch_size 256 \
  --use_subject_vote \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --classify_choice multi_class \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15


# CNBPM
python -u run.py \
  --method LaBraM \
  --checkpoints_path ./checkpoints/LaBraM/pretrain_labram/LaBraM/labram-base.pth \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-CNBPM-Multi \
  --model LaBraM \
  --data MultiDatasets \
  --training_dataset CNBPM \
  --sampling_rate_list 200 \
  --batch_size 256 \
  --use_subject_vote \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --classify_choice multi_class \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15


## APAVA
python -u run.py \
  --method LaBraM \
  --checkpoints_path ./checkpoints/LaBraM/pretrain_labram/LaBraM/labram-base.pth \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-APAVA-Multi \
  --model LaBraM \
  --data MultiDatasets \
  --training_dataset APAVA \
  --sampling_rate_list 200 \
  --batch_size 128 \
  --use_subject_vote \
  --classify_choice multi_class \
  --swa \
  --cross_val fixed \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# ADFSU
python -u run.py \
  --method LaBraM \
  --checkpoints_path ./checkpoints/LaBraM/pretrain_labram/LaBraM/labram-base.pth \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFSU-Multi \
  --model LaBraM \
  --data MultiDatasets \
  --training_dataset ADFSU \
  --batch_size 128 \
  --sampling_rate_list 100 \
  --use_subject_vote \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --classify_choice multi_class \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# ADSZ
python -u run.py \
  --method LaBraM \
  --checkpoints_path ./checkpoints/LaBraM/pretrain_labram/LaBraM/labram-base.pth \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADSZ-Multi \
  --model LaBraM \
  --data MultiDatasets \
  --training_dataset ADSZ \
  --batch_size 128 \
  --sampling_rate_list 100 \
  --use_subject_vote \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --classify_choice multi_class \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15
