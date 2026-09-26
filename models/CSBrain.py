import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F

from layers.CSBrain_Layer import *


TOPOLOGY = {
    0: ['Fp1', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Fp2'],
    1: ['P3', 'Pz', 'P4'],
    2: ['P7', 'T7', 'T8', 'P8'],
    3: ['O1', 'O2'],
    4: ['C3', 'Cz', 'C4']
}


def build_csbrain_channel_layout(channel_names, brain_regions):
    """Validate meta-driven channel order and build CSBrain sorted indices.

    ``brain_regions`` remains user/config controlled to preserve the original
    CSBrain setup. Channel names themselves come from meta.json. Legacy 10-20
    temporal names are normalized only for topology lookup.
    """
    if len(channel_names) != len(brain_regions):
        raise ValueError(
            f"CSBrain requires one brain-region label per channel: "
            f"got {len(channel_names)} channels and {len(brain_regions)} region labels."
        )

    legacy_alias = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    region_groups = {}
    for index, (channel, region) in enumerate(zip(channel_names, brain_regions)):
        if region not in TOPOLOGY:
            raise ValueError(
                f"CSBrain region {region} is undefined. Available regions: {sorted(TOPOLOGY)}."
            )
        canonical_channel = legacy_alias.get(channel, channel)
        if canonical_channel not in TOPOLOGY[region]:
            raise ValueError(
                f"CSBrain channel '{channel}' (mapped to '{canonical_channel}') is not "
                f"listed in topology region {region}. Check --brain_regions against "
                f"meta.json CHANNELS order."
            )
        region_groups.setdefault(region, []).append((index, canonical_channel))

    sorted_indices = []
    for region in sorted(region_groups):
        region_electrodes = sorted(
            region_groups[region], key=lambda item: TOPOLOGY[region].index(item[1])
        )
        sorted_indices.extend(index for index, _ in region_electrodes)
    return sorted_indices


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        """electrode_labels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
            'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
        ]"""
        channel_names_by_id = dict(getattr(configs, "channel_names_by_id", {}))
        if len(channel_names_by_id) != 1:
            raise ValueError(
                "CSBrain downstream evaluation expects exactly one dataset so channel names "
                "can be read automatically from meta.json."
            )
        channel_names = list(next(iter(channel_names_by_id.values())))
        if len(channel_names) != configs.enc_in:
            raise ValueError("meta.json CHANNELS length does not match enc_in for CSBrain")
        # Brain region encoding
        # brain_regions = [0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4, 2, 2, 1, 1, 1, 2, 3, 3]
        brain_regions = list(map(int, configs.brain_regions.split(",")))
        if len(brain_regions) != configs.enc_in:
            raise ValueError(
                "CSBrain --brain_regions must contain exactly one region label for each "
                "channel in meta.json CHANNELS (in the same order)."
            )
        sorted_indices = build_csbrain_channel_layout(channel_names, brain_regions)
        print("CSBrain channel order from meta.json:", channel_names)
        print("CSBrain sorted indices:", sorted_indices)

        self.backbone = CSBrain(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8,
            brain_regions=brain_regions,
            sorted_indices=sorted_indices
        )
        self.task_name = configs.task_name
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = configs.checkpoints_path
        # since the author of CBraMod does not use swa, we directly load the model here for simplicity
        if os.path.exists(model_path) and configs.is_training == 1 and configs.task_name == "supervised":
            state_dict = torch.load(model_path, map_location=device)
            # Remove "module." prefix
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            model_state_dict = self.backbone.state_dict()
            # Filter matching weights by shape
            matching_dict = {k: v for k, v in new_state_dict.items() if
                             k in model_state_dict and v.size() == model_state_dict[k].size()}
            model_state_dict.update(matching_dict)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(model_state_dict)
            print('Missing keys:', missing_keys)
            print('Unexpected keys:', unexpected_keys)
            print(f"Loading model successful at {model_path}")
        self.backbone.proj_out = nn.Identity()
        self.duration = math.ceil(configs.seq_len / 200)  # duration in seconds
        self.classifier = nn.Sequential(
            nn.Linear(int(configs.enc_in*self.duration*200), 4*200),
            nn.ELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(4*200, 200),
            nn.ELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(200, configs.num_class)
        )

    def pad_to_multiple(self, x: torch.Tensor, multiple: int = 200):
        '''
        Zero-pad sequence length (dim=1) so that it's a multiple of `multiple`.
        Input shape: (batch_size, seq_len, feature_dim)
        '''
        seq_len = x.size(1)
        remainder = seq_len % multiple
        if remainder != 0:
            pad_len = multiple - remainder
            x = F.pad(x, (0, 0, 0, pad_len))  # pad along seq_length dim
        return x

    def supervised(self, x_enc, label_id=None):  # x_enc (batch_size, seq_length, enc_in)
        # padding and channel mapping for loading CBraMod weights
        x_enc = self.pad_to_multiple(x_enc, multiple=200).permute(0, 2, 1)  # pad to multiple of 200
        x_enc = x_enc.view(x_enc.size(0), x_enc.size(1), self.duration, 200)
        batch_size, enc_in, duration, patch_length = x_enc.shape
        feats = self.backbone(x_enc)
        feats = feats.contiguous().view(batch_size, enc_in * duration * patch_length)
        out = self.classifier(feats)
        return out

    def forward(self, x_enc, label_id=None, mask=None, **kwargs):
        if self.task_name == "supervised" or self.task_name == "finetune":
            output = self.supervised(x_enc, label_id=label_id)
            return output
        else:
            raise ValueError("Task name not recognized or not implemented within the CSBrain model")
