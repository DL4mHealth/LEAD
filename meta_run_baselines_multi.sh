#!/bin/bash

# Run scripts sequentially

# BIOT
bash ./scripts/BIOT/supervised/BIOT/S-1-Multi.sh

# CBraMod
bash ./scripts/CBraMod/supervised/CBraMod/S-1-Multi.sh

# LaBraM
bash ./scripts/LaBraM/supervised/LaBraM/S-1-Multi.sh

# CSBrain
bash ./scripts/CSBrain/supervised/CSBrain/S-1-Multi.sh

# EEGConformer
bash scripts/EEGConformer/supervised/EEGConformer/S-1-Multi.sh

# EEGInception
bash scripts/EEGInception/supervised/EEGInception/S-1-Multi.sh

# EEGNet
bash scripts/EEGNet/supervised/EEGNet/S-1-Multi.sh

# iTransformer
bash scripts/iTransformer/supervised/iTransformer/S-1-Multi.sh

# ManualFeature
bash scripts/ManualFeature/supervised/ManualFeature/S-1-Multi.sh

# Medformer
bash scripts/Medformer/supervised/ADformer/S-1-Multi.sh

# MedGNN
bash scripts/MedGNN/supervised/MedGNN/S-1-Multi.sh

# MNet
bash scripts/MNet/supervised/MNet/S-1-Multi.sh

# ModernTCN
bash scripts/ModernTCN/supervised/ModernTCN/S-1-Multi.sh

# PatchTST
bash scripts/PatchTST/supervised/PatchTST/S-1-Multi.sh

# TCN
bash scripts/TCN/supervised/TCN/S-1-Multi.sh

# TimesNet
bash scripts/TimesNet/supervised/TimesNet/S-1-Multi.sh
