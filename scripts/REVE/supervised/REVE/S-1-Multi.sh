#!/usr/bin/env bash

# Multi-Class Classification

# ADFTD
python -u run.py \
  --method REVE \
  --checkpoints_path hf:brain-bzh/reve-base \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFTD-Multi \
  --model REVE \
  --data MultiDatasets \
  --training_dataset ADFTD \
  --sampling_rate_list 200 \
  --batch_size 512 \
  --classify_choice multi_class \
  --use_subject_vote \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# CNBPM
python -u run.py \
  --method REVE \
  --checkpoints_path hf:brain-bzh/reve-base \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-CNBPM-Multi \
  --model REVE \
  --data MultiDatasets \
  --training_dataset CNBPM \
  --sampling_rate_list 200 \
  --batch_size 512 \
  --classify_choice multi_class \
  --use_subject_vote \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# APAVA
python -u run.py \
  --method REVE \
  --checkpoints_path hf:brain-bzh/reve-base \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-APAVA-Multi \
  --model REVE \
  --data MultiDatasets \
  --training_dataset APAVA \
  --sampling_rate_list 200 \
  --batch_size 128 \
  --classify_choice multi_class \
  --use_subject_vote \
  --cross_val fixed \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# ADFSU
# REVE was pretrained/released for 200 Hz EEG. This command is intentionally disabled
# because the legacy baseline setting for this dataset uses 100 Hz.
# python -u run.py \
#   --method REVE \
#   --checkpoints_path hf:brain-bzh/reve-base \
#   --task_name supervised \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --model_id S-ADFSU-Multi \
#   --model REVE \
#   --data MultiDatasets \
#   --training_dataset ADFSU \
#   --sampling_rate_list 100 \
#   --batch_size 128 \
#   --classify_choice multi_class \
#   --use_subject_vote \
#   --ratio_a 0.8 \
#   --ratio_b 0.9 \
#   --swa \
#   --des Exp \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 15

# ADSZ
# REVE was pretrained/released for 200 Hz EEG. This command is intentionally disabled
# because the legacy baseline setting for this dataset uses 100 Hz.
# python -u run.py \
#   --method REVE \
#   --checkpoints_path hf:brain-bzh/reve-base \
#   --task_name supervised \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --model_id S-ADSZ-Multi \
#   --model REVE \
#   --data MultiDatasets \
#   --training_dataset ADSZ \
#   --sampling_rate_list 100 \
#   --batch_size 128 \
#   --classify_choice multi_class \
#   --use_subject_vote \
#   --ratio_a 0.8 \
#   --ratio_b 0.9 \
#   --swa \
#   --des Exp \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 15
