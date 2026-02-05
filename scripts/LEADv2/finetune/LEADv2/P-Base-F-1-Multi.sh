export CUDA_VISIBLE_DEVICES=0,1,2,3

# Finetuning


# Multi-Class Classification
# ADFTD
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L400/ --model_id P-Base-F-ADFTD-Multi --model LEADv2 --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations flip,frequency,jitter,mask,channel,drop --patch_len 50 --stride 50 --group_shuffle --group_size 8 --use_subject_loss --sampling_rate_list 200,100,50 \
--ratio_a 0.8 --ratio_b 0.9 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --use_subject_vote --swa \
--classify_choice multi_class --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L400/ --model_id P-Base-F-CNBPM-Multi --model LEADv2 --data MultiDatasets \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations frequency,jitter,mask,drop --patch_len 50 --stride 25 --dropout 0.25 --group_shuffle --group_size 4 --use_subject_loss --sampling_rate_list 200,100,50 \
--ratio_a 0.8 --ratio_b 0.9 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --use_subject_vote --swa \
--classify_choice multi_class --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# APAVA
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L200/ --model_id P-Base-F-APAVA-Multi --model LEADv2 --data MultiDatasets \
--training_datasets APAVA --testing_datasets APAVA \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations patch --patch_len 50 --stride 50 --group_shuffle --group_size 2 --use_subject_loss \
--sampling_rate_list 200,100,50 --montage_name standard_1005 --channel_names C3,C4,F3,F4,F7,F8,Fp1,Fp2,O1,O2,P3,P4,T7,T8,P7,P8 --use_subject_vote \
--swa --classify_choice multi_class --cross_val fixed --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ADFSU
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L100/ --model_id P-Base-F-ADFSU-Multi --model LEADv2 --data MultiDatasets \
--training_datasets ADFSU --testing_datasets ADFSU \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations flip,frequency,jitter,mask,patch,channel,drop --patch_len 50 --stride 50 --group_shuffle --group_size 2 --use_subject_loss \
--sampling_rate_list 100,50 --ratio_a 0.8 --ratio_b 0.9 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --use_subject_vote \
--swa --classify_choice multi_class --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
