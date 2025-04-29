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


def augment_batch(batch_X, mask_prob=0.1):
    mask = torch.rand_like(batch_X.float()) < mask_prob
    # Replace with "unknown" token (assuming 0 is unused/unknown)
    return torch.where(mask, torch.zeros_like(batch_X), batch_X)


def train(model, X, Y, epochs=15, batch_size=32):
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X.size()[0])
        total_loss = 0

        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_Y = X[indices], Y[indices]
            batch_X = augment_batch(batch_X)

            optimizer.zero_grad()
            outputs = model(batch_X)

            losses = [
                criterions[j](outputs[j], batch_Y[:, j])
                for j in range(len(output_cols))
            ]
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(X)}')