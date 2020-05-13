import math
import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Transformer


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim,emb_dim,enc_hid_dim,dec_hid_dim,dropout=0.5):
        
        super(BiLSTMEncoder,self).__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_dim,emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear( enc_hid_dim * 2, dec_hid_dim )
        
        self.dropout = nn.Dropout( dropout )
        
    def forward(self, X):
        
        embedded = self.dropout(self.embedding(X))
        outputs, hidden = self.rnn(embedded)
        hidden = F.tanh( self.fc ( torch.cat( (hidden[-2,:,:], hidden[-1, : , : ] ), dim = 1 ) ) )

        return outputs, hidden


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

    def __init__(self, nhead, num_encoder_layers, num_decoder_layers, \
                    dim_feedforward, max_seq_length,vocab_size, pad_idx, \
                        d_model=None, pos_dropout =0.1, trans_dropout= 0.1, embeddings=None):
        super().__init__()
       
        if embeddings is None:
            self.embed_src = nn.Embedding(vocab_size, d_model)
            self.embed_tgt = nn.Embedding(vocab_size, d_model)
        else:
            d_model = embeddings.size(1)
            self.d_model = embeddings.size(1)
            self.embed_src = nn.Embedding(*embeddings.shape)
            self.embed_src.weight = nn.Parameter(embeddings, requires_grad=False)
            
            self.embed_tgt = nn.Embedding(*embeddings.shape)
            self.embed_tgt.weight = nn.Parameter(embeddings, requires_grad=False)
        
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        
        self.fc = nn.Linear(d_model, vocab_size)

        self.pad_idx = pad_idx
        
        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None


    def generate_square_mask(self,sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def make_pad_mask(self,seq,pad_idx):
        mask = (seq == pad_idx).transpose(0,1)
        return mask

    def forward(self, src, tgt):

        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            self.tgt_mask = self.generate_square_mask(len(tgt)).to(tgt.device)
        
        src_pad_mask = self.make_pad_mask(src,self.pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt,self.pad_idx)

        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(src, tgt, src_mask=self.src_mask, tgt_mask=self.tgt_mask, memory_mask=self.memory_mask, 
                                    src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
        
        return self.fc(output)