import time
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=16,
        n_fft=200,
        hop_length=100,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.register_buffer('index', torch.arange(n_channels, dtype=torch.long), persistent=True)

    def stft(self, sample):
        spectral = torch.stft(
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # perturb
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb


# supervised classifier module
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x


"""if __name__ == "__main__":
    x = torch.randn(16, 18, 200)
    model = BIOTClassifier(
        emb_size=256,
        heads=8,
        depth=4,
        n_classes=2,
        n_fft=200,
        hop_length=100,
        n_channels=18,  # here is 18
    )
    out = model(x)
    print(out.shape)"""


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.channel_mapping = nn.Conv1d(
            in_channels=configs.enc_in,
            out_channels=18,  # default 18 in BIOT checkpoint pretrained on six dataset
            kernel_size=1,
            padding='same',
            bias=False,
        )
        self.model = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=configs.num_class,
            n_fft=200,
            hop_length=100,
            n_channels=18,  # here is 18 in their pretrained model
        )
        self.task_name = configs.task_name
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = configs.checkpoints_path
        if os.path.exists(model_path) and configs.is_training == 1 and configs.task_name == "supervised":
            missing_keys, unexpected_keys = self.model.biot.load_state_dict(torch.load(model_path, map_location=device))
            print('Missing keys:', missing_keys)
            print('Unexpected keys:', unexpected_keys)
            print(f"Loading model successful at {model_path}")

    def supervised(self, x_enc, x_mark_enc):  # x_enc (batch_size, seq_length, enc_in)
        if x_enc.size(1) < 200:  # some dataset may be shorter than 200 time points
            pad_len = 200 - x_enc.size(1)
            # pad on the second last dimension right side
            x_enc = F.pad(x_enc, (0, 0, 0, pad_len))
        # Channel mapping to 128
        enc_out = self.channel_mapping(x_enc.permute(0, 2, 1))  # (batch_size, 16, seq_length)
        out = self.model(enc_out)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, fs=None, mask=None):
        if self.task_name == "supervised":
            output = self.supervised(x_enc, x_mark_enc)
            return output
        else:
            raise ValueError("Task name not recognized or not implemented within the BIOT model")
