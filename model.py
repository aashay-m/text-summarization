import math
import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerSummarizer(nn.Module):
    def __init__(self, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length,vocab_size,d_model=None, pos_dropout =0.1, trans_dropout= 0.1,embeddings=None):
        super().__init__()
       
        if embeddings is None:
            self.embed_src = nn.Embedding(vocab_size, d_model)
            self.embed_tgt = nn.Embedding(vocab_size, d_model)
        else:
            d_model = embeddings.size(1)
            self.d_model = embeddings.size(1)
            self.embed_src = nn.Embedding(*embeddings.shape)
            self.embed_src.weight = nn.Parameter(embeddings,requires_grad=False)
            
            self.embed_tgt = nn.Embedding(*embeddings.shape)
            self.embed_tgt.weight = nn.Parameter(embeddings,requires_grad=False)
        
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        
#         print("Before Embed: ",src.shape,tgt.shape,sep="\n")
        
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
#         print(src.shape)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
#         print(tgt.shape)

        output = self.transformer(src, tgt)
        
        return self.fc(output)