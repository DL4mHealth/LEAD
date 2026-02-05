export CUDA_VISIBLE_DEVICES=0,1,2,3

# Multi-Class Classification
# ADFTD
python -u run.py --method CSBrain --checkpoints_path ./checkpoints/CSBrain/pretrain_csbrain/CSBrain/CSBrain.pth \
--task_name supervised --is_training 1 --root_path ./dataset/L400/ --model_id S-ADFTD-Multi --model CSBrain --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--batch_size 512 --use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --classify_choice multi_class --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# CNBPM
python -u run.py --method CSBrain --checkpoints_path ./checkpoints/CSBrain/pretrain_csbrain/CSBrain/CSBrain.pth \
--task_name supervised --is_training 1 --root_path ./dataset/L400/ --model_id S-CNBPM-Multi --model CSBrain --data MultiDatasets \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--batch_size 512 --use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --classify_choice multi_class --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


## APAVA
python -u run.py --method CSBrain --checkpoints_path ./checkpoints/CSBrain/pretrain_csbrain/CSBrain/CSBrain.pth \
--task_name supervised --is_training 1 --root_path ./dataset/L200/ --model_id S-APAVA-Multi --model CSBrain --data MultiDatasets \
--training_datasets APAVA \
--testing_datasets APAVA \
--batch_size 128 --use_subject_vote --classify_choice multi_class --channel_names C3,C4,F3,F4,F7,F8,Fp1,Fp2,O1,O2,P3,P4,T7,T8,P7,P8 --brain_regions 4,4,0,0,0,0,0,0,3,3,1,1,2,2,2,2 --swa \
--cross_val fixed --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ADFSU
python -u run.py --method CSBrain --checkpoints_path ./checkpoints/CSBrain/pretrain_csbrain/CSBrain/CSBrain.pth \
--task_name supervised --is_training 1 --root_path ./dataset/L100/ --model_id S-ADFSU-Multi --model CSBrain --data MultiDatasets \
--training_datasets ADFSU \
--testing_datasets ADFSU \
--batch_size 128 --sampling_rate_list 100 --use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --classify_choice multi_class --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ADSZ
python -u run.py --method CSBrain --checkpoints_path ./checkpoints/CSBrain/pretrain_csbrain/CSBrain/CSBrain.pth \
--task_name supervised --is_training 1 --root_path ./dataset/L100/ --model_id S-ADSZ-Multi --model CSBrain --data MultiDatasets \
--training_datasets ADSZ \
--testing_datasets ADSZ \
--batch_size 128 --sampling_rate_list 100 --use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --classify_choice multi_class --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15