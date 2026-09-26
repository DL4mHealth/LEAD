#!/usr/bin/env bash


# Multi-class classification
# ADFTD
python -u run.py \
  --method TimesNet \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFTD-Multi \
  --model TimesNet \
  --data MultiDatasets \
  --training_dataset ADFTD \
  --sampling_rate_list 200 \
  --e_layers 2 \
  --batch_size 512 \
  --top_k 3 \
  --d_model 32 \
  --d_ff 64 \
  --classify_choice multi_class \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --use_subject_vote \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# CNBPM
python -u run.py \
  --method TimesNet \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-CNBPM-Multi \
  --model TimesNet \
  --data MultiDatasets \
  --training_dataset CNBPM \
  --sampling_rate_list 200 \
  --e_layers 2 \
  --batch_size 512 \
  --top_k 3 \
  --d_model 32 \
  --d_ff 64 \
  --classify_choice multi_class \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --use_subject_vote \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

## APAVA
python -u run.py \
  --method TimesNet \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-APAVA-Multi \
  --model TimesNet \
  --data MultiDatasets \
  --training_dataset APAVA \
  --sampling_rate_list 200 \
  --e_layers 2 \
  --batch_size 128 \
  --top_k 3 \
  --d_model 32 \
  --d_ff 64 \
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
  --method TimesNet \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFSU-Multi \
  --model TimesNet \
  --data MultiDatasets \
  --training_dataset ADFSU \
  --e_layers 2 \
  --batch_size 128 \
  --top_k 3 \
  --d_model 32 \
  --d_ff 64 \
  --classify_choice multi_class \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --use_subject_vote \
  --swa \
  --sampling_rate_list 100 \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# ADSZ
python -u run.py \
  --method TimesNet \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADSZ-Multi \
  --model TimesNet \
  --data MultiDatasets \
  --training_dataset ADSZ \
  --e_layers 2 \
  --batch_size 128 \
  --top_k 3 \
  --d_model 32 \
  --d_ff 64 \
  --classify_choice multi_class \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --use_subject_vote \
  --swa \
  --sampling_rate_list 100 \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15
