#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Run scripts sequentially


# supervised
bash ./scripts/LEAD/supervised/LEAD/S-1-Multi.sh
# probe (make sure you have pretraining checkpoint for following scripts)
bash ./scripts/LEAD/probe/LEAD/P-Base-Pr-1-Multi.sh
# paradigm comparison
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi-Paradigm.sh
# leave-one-subject-out cross-validation
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi-Loso.sh
