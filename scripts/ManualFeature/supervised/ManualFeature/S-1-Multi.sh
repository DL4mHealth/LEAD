#!/usr/bin/env bash


# Multi-class classification
# ADFTD
python -u run.py \
  --method ManualFeature \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFTD-Multi \
  --model ManualFeature \
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
  --method ManualFeature \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-CNBPM-Multi \
  --model ManualFeature \
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

## APAVA
python -u run.py \
  --method ManualFeature \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-APAVA-Multi \
  --model ManualFeature \
  --data MultiDatasets \
  --training_dataset APAVA \
  --sampling_rate_list 200 \
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
  --method ManualFeature \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFSU-Multi \
  --model ManualFeature \
  --data MultiDatasets \
  --training_dataset ADFSU \
  --batch_size 128 \
  --sampling_rate_list 100 \
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

# ADSZ
python -u run.py \
  --method ManualFeature \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADSZ-Multi \
  --model ManualFeature \
  --data MultiDatasets \
  --training_dataset ADSZ \
  --batch_size 128 \
  --sampling_rate_list 100 \
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
