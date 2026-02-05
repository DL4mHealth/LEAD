import torch
import torch.nn as nn
import math

from layers.CSBrain_Layer import *


"""class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        electrode_labels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
        ]

        # Brain region encoding
        brain_regions = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1]

        # Topological structure
        topology = {
            0: ['Fp1', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Fp2'],
            1: ['P3', 'Pz', 'P4'],
            2: ['T5', 'T3', 'T4', 'T6'],
            3: ['O1', 'O2'],
            4: ['C3', 'Cz', 'C4']
        }

        # Group electrode indices by brain region
        region_groups = {}
        for i, region in enumerate(brain_regions):
            if region not in region_groups:
                region_groups[region] = []
            region_groups[region].append((i, electrode_labels[i]))

        # Sort based on topology
        sorted_indices = []
        for region in sorted(region_groups.keys()):
            region_electrodes = region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
            sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", sorted_indices)

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=brain_regions,
                sorted_indices=sorted_indices
            )
        else:
            return 0

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            # Remove "module." prefix
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

            model_state_dict = self.backbone.state_dict()

            # Filter matching weights by shape
            matching_dict = {k: v for k, v in new_state_dict.items() if
                             k in model_state_dict and v.size() == model_state_dict[k].size()}

            model_state_dict.update(matching_dict)
            self.backbone.load_state_dict(model_state_dict)

        self.backbone.proj_out = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(19 * 5 * 200, 5 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(5 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * 200)
        print(feats.shape)
        out = self.classifier(feats)
        out = out[:, 0]
        return out


if __name__ == "__main__":
    class Param:
        def __init__(self):
            self.model = 'CSBrain'
            self.n_layer = 12
            self.use_pretrained_weights = True
            self.foundation_dir = '../checkpoints/CSBrain/pretrain_csbrain/CSBrain/CSBrain.pth'
            self.cuda = 0
            self.dropout = 0.1

    param = Param()
    model = Model(param)
    x = torch.randn(8, 19, 5, 200)  # Example input
    out = model(x)
    print(out.shape)  # Should print torch.Size([2])"""

TOPOLOGY = {
    0: ['Fp1', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Fp2'],
    1: ['P3', 'Pz', 'P4'],
    2: ['P7', 'T7', 'T8', 'P8'],
    3: ['O1', 'O2'],
    4: ['C3', 'Cz', 'C4']
}  # You need to manually define the topology for each brain region of different datasets


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        """electrode_labels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
            'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
        ]"""
        channel_names = configs.channel_names.split(",")
        if len(channel_names) != configs.enc_in:
            raise ValueError("channel_names length does not match enc_in")
        # Brain region encoding
        # brain_regions = [0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4, 2, 2, 1, 1, 1, 2, 3, 3]
        brain_regions = list(map(int, configs.brain_regions.split(",")))
        if len(brain_regions) != configs.enc_in:
            raise ValueError("brain_regions length does not match enc_in")
        # Group electrode indices by brain region
        region_groups = {}
        for i, region in enumerate(brain_regions):
            if region not in region_groups:
                region_groups[region] = []
            region_groups[region].append((i, channel_names[i]))

        # Sort based on topology
        sorted_indices = []
        for region in sorted(region_groups.keys()):
            region_electrodes = region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: TOPOLOGY[region].index(x[1]))
            sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", sorted_indices)

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

    def supervised(self, x_enc, x_mark_enc):  # x_enc (batch_size, seq_length, enc_in)
        # padding and channel mapping for loading CBraMod weights
        x_enc = self.pad_to_multiple(x_enc, multiple=200).permute(0, 2, 1)  # pad to multiple of 200
        x_enc = x_enc.view(x_enc.size(0), x_enc.size(1), self.duration, 200)
        batch_size, enc_in, duration, patch_length = x_enc.shape
        feats = self.backbone(x_enc)
        feats = feats.contiguous().view(batch_size, enc_in * duration * patch_length)
        out = self.classifier(feats)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            output = self.supervised(x_enc, x_mark_enc)
            return output
        else:
            raise ValueError("Task name not recognized or not implemented within the LEAD model")
