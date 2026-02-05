export CUDA_VISIBLE_DEVICES=0,1,2,3


# Multi-class classification
# ADFTD
python -u run.py --method Medformer --task_name supervised \
--is_training 1 --root_path ./dataset/L400/ --model_id S-ADFTD-Multi --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 5,10,20 --no_channel_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --ratio_a 0.8 --ratio_b 0.9 --use_subject_vote --classify_choice multi_class --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method Medformer --task_name supervised \
--is_training 1 --root_path ./dataset/L400/ --model_id S-CNBPM-Multi --model ADformer --data MultiDatasets \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--patch_len_list 5,10,20 --no_channel_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --ratio_a 0.8 --ratio_b 0.9 --use_subject_vote --classify_choice multi_class --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

## APAVA
python -u run.py --method Medformer --task_name supervised \
--is_training 1 --root_path ./dataset/L200/ --model_id S-APAVA-Multi --model ADformer --data MultiDatasets \
--training_datasets APAVA \
--testing_datasets APAVA \
--patch_len_list 5,10,20 --no_channel_block \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --use_subject_vote --classify_choice multi_class --swa \
--cross_val fixed --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADFSU
python -u run.py --method Medformer --task_name supervised \
--is_training 1 --root_path ./dataset/L100/ --model_id S-ADFSU-Multi --model ADformer --data MultiDatasets \
--training_datasets ADFSU \
--testing_datasets ADFSU \
--patch_len_list 5,10,20 --no_channel_block \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --sampling_rate_list 100 --ratio_a 0.8 --ratio_b 0.9 --use_subject_vote --classify_choice multi_class --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADSZ
python -u run.py --method Medformer --task_name supervised \
--is_training 1 --root_path ./dataset/L100/ --model_id S-ADSZ-Multi --model ADformer --data MultiDatasets \
--training_datasets ADSZ \
--testing_datasets ADSZ \
--patch_len_list 5,10,20 --no_channel_block \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --sampling_rate_list 100 --ratio_a 0.8 --ratio_b 0.9 --use_subject_vote --classify_choice multi_class --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15