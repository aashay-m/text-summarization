import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,X,y):
        super().__init__()
        n = len(self.X)
        # BiLSTM layer for the encoder
        bilstm = nn.LSTM(n,n,bidirectional=True)
    def encode():
        F.relu()
