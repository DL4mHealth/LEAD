export CUDA_VISIBLE_DEVICES=0,1,2,3


# Multi-class classification
# ADFTD
python -u run.py --method ManualFeature --task_name supervised \
--is_training 1 --root_path ./dataset/L400/ --model_id S-ADFTD-Multi --model ManualFeature --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--batch_size 512 --classify_choice multi_class --use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method ManualFeature --task_name supervised \
--is_training 1 --root_path ./dataset/L400/ --model_id S-CNBPM-Multi --model ManualFeature --data MultiDatasets \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--batch_size 512 --classify_choice multi_class --use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

## APAVA
python -u run.py --method ManualFeature --task_name supervised \
--is_training 1 --root_path ./dataset/L200/ --model_id S-APAVA-Multi --model ManualFeature --data MultiDatasets \
--training_datasets APAVA \
--testing_datasets APAVA \
--batch_size 128 --classify_choice multi_class --use_subject_vote --swa \
--cross_val fixed --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADFSU
python -u run.py --method ManualFeature --task_name supervised \
--is_training 1 --root_path ./dataset/L100/ --model_id S-ADFSU-Multi --model ManualFeature --data MultiDatasets \
--training_datasets ADFSU \
--testing_datasets ADFSU \
--batch_size 128 --sampling_rate_list 100 --classify_choice multi_class --use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADSZ
python -u run.py --method ManualFeature --task_name supervised \
--is_training 1 --root_path ./dataset/L100/ --model_id S-ADSZ-Multi --model ManualFeature --data MultiDatasets \
--training_datasets ADSZ \
--testing_datasets ADSZ \
--batch_size 128 --sampling_rate_list 100 --classify_choice multi_class --use_subject_vote --ratio_a 0.8 --ratio_b 0.9 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15