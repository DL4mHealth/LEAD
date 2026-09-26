#!/usr/bin/env bash

# Multi-Class Classification

# ADFTD
python -u run.py \
  --method EEGDeformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFTD-Multi \
  --model EEGDeformer \
  --data MultiDatasets \
  --training_dataset ADFTD \
  --sampling_rate_list 200 \
  --e_layers 4 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 16 \
  --dropout 0.5 \
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
  --method EEGDeformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-CNBPM-Multi \
  --model EEGDeformer \
  --data MultiDatasets \
  --training_dataset CNBPM \
  --sampling_rate_list 200 \
  --e_layers 4 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 16 \
  --dropout 0.5 \
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
  --method EEGDeformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-APAVA-Multi \
  --model EEGDeformer \
  --data MultiDatasets \
  --training_dataset APAVA \
  --sampling_rate_list 200 \
  --e_layers 4 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 16 \
  --dropout 0.5 \
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
python -u run.py \
  --method EEGDeformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFSU-Multi \
  --model EEGDeformer \
  --data MultiDatasets \
  --training_dataset ADFSU \
  --sampling_rate_list 100 \
  --e_layers 4 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 16 \
  --dropout 0.5 \
  --batch_size 128 \
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
  --method EEGDeformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADSZ-Multi \
  --model EEGDeformer \
  --data MultiDatasets \
  --training_dataset ADSZ \
  --sampling_rate_list 100 \
  --e_layers 4 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 16 \
  --dropout 0.5 \
  --batch_size 128 \
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
