#
# Transformer for nanopore
#

#
# Transformer model
#

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NanoporeTransformer(nn.Module):

    def __init__(self,
                 n_head=8,
                 n_layers=6,
                 d_model=128,
                 dim_feedforward=512,
                 dropout=0.1):
        super(NanoporeTransformer, self).__init__()

        self.linear_encoder = nn.Linear(1, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_model = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = (x - 90.) / 40.
        x = self.linear_encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_model(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x
