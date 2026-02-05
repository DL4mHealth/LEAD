#!/bin/bash

# Run scripts sequentially

# LEADv2
bash ./scripts/LEADv2/pretrain_lead/LEADv2/P-Base.sh
bash ./scripts/LEADv2/finetune/LEADv2/P-Base-F-1-Multi.sh

