#!/usr/bin/env bash


# Multi-class classification
# ADFTD
python -u run.py \
  --method Medformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFTD-Multi \
  --model ADformer \
  --data MultiDatasets \
  --training_dataset ADFTD \
  --sampling_rate_list 200 \
  --patch_len_list 5,10,20 \
  --no_channel_block \
  --e_layers 12 \
  --batch_size 512 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff 256 \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --use_subject_vote \
  --classify_choice multi_class \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# CNBPM
python -u run.py \
  --method Medformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-CNBPM-Multi \
  --model ADformer \
  --data MultiDatasets \
  --training_dataset CNBPM \
  --sampling_rate_list 200 \
  --patch_len_list 5,10,20 \
  --no_channel_block \
  --e_layers 12 \
  --batch_size 512 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff 256 \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --use_subject_vote \
  --classify_choice multi_class \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

## APAVA
python -u run.py \
  --method Medformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-APAVA-Multi \
  --model ADformer \
  --data MultiDatasets \
  --training_dataset APAVA \
  --sampling_rate_list 200 \
  --patch_len_list 5,10,20 \
  --no_channel_block \
  --e_layers 12 \
  --batch_size 128 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff 256 \
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
  --method Medformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADFSU-Multi \
  --model ADformer \
  --data MultiDatasets \
  --training_dataset ADFSU \
  --patch_len_list 5,10,20 \
  --no_channel_block \
  --e_layers 12 \
  --batch_size 128 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff 256 \
  --sampling_rate_list 100 \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --use_subject_vote \
  --classify_choice multi_class \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15

# ADSZ
python -u run.py \
  --method Medformer \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id S-ADSZ-Multi \
  --model ADformer \
  --data MultiDatasets \
  --training_dataset ADSZ \
  --patch_len_list 5,10,20 \
  --no_channel_block \
  --e_layers 12 \
  --batch_size 128 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff 256 \
  --sampling_rate_list 100 \
  --ratio_a 0.8 \
  --ratio_b 0.9 \
  --use_subject_vote \
  --classify_choice multi_class \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 15
