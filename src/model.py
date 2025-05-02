import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os


class TabTransformer(nn.Module):
    def __init__(self, category_sizes, dim, output_sizes):
        super().__init__()
        # Embeddings for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim) for size in category_sizes
        ])

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4, 
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True  
            ),
            num_layers=3,
            enable_nested_tensor=False  # Suppress warning
        )

        # Output heads with layer normalization
        self.heads = nn.ModuleList()
        for out_size in output_sizes:
            head = nn.Sequential(
                nn.LayerNorm(dim * len(category_sizes)),
                nn.Linear(dim * len(category_sizes), 64),
                nn.ReLU(),
                nn.Linear(64, out_size)
            )
            self.heads.append(head)

    def forward(self, x):
        # Embed each feature
        x_emb = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.stack(x_emb, dim=1)  # [batch, features, dim]

        # Transformer processing
        x_trans = self.transformer(x_emb)

        # Flatten for heads
        x_flat = x_trans.flatten(start_dim=1)

        # Multiple outputs
        return [head(x_flat) for head in self.heads]
    

class ImprovedTabTransformer(nn.Module):
    def __init__(self, category_sizes, dim, output_sizes):
        super().__init__()
        # Add feature-wise projections

        # self.embeddings = nn.ModuleList([
        #   nn.Sequential(
        #        nn.Embedding(size, dim),
        #        nn.LayerNorm(dim)
        #    ) for size in category_sizes
        #])

        self.embeddings = nn.ModuleList([nn.Embedding(size, dim) for size in category_sizes])
        self.column_embeddings = nn.Parameter(torch.randn(len(category_sizes), dim))

        # Add learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,  # More attention heads
                dim_feedforward=256,
                dropout=0.4,
                batch_first=True
            ),
            num_layers=4
        )

        # Deeper output heads
        self.heads = nn.ModuleList()
        for out_size in output_sizes:
            head = nn.Sequential(
                    nn.Linear(dim, dim),         
                    nn.GELU(),
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim//2),
                    nn.GELU(),
                    nn.LayerNorm(dim//2),
                    nn.Linear(dim//2, out_size)
            )
            self.heads.append(head)

    def forward(self, x):
        # x_emb = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = [self.embeddings[i](x[:, i]) + self.column_embeddings[i] for i in range(len(self.embeddings))]
        x_emb = torch.stack(x_emb, dim=1)  # [batch, features, dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(x_emb.size(0), -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)

        x_trans = self.transformer(x_emb)

        # Use CLS token for prediction
        cls_output = x_trans[:, 0]

        return [head(cls_output) for head in self.heads]