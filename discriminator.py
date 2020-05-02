import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CNNDiscriminator(nn.Module):
    def __init__(self,edim,vocab_size,filter_sizes,num_filters,padding_idx,gpu_flag=False,droupout=0.2,initialize='normal'):
        super(CNNDiscriminator,self).__init__()
        self.embedding_dim = edim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        self.gpu_flag = gpu_flag

        self.embeddings = nn.Embedding(vocab_size,edim,padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1,n,(f,edim)) for (n,f) in zip(num_filters,filter_sizes)
            ])
        self.fc = nn.Linear(self.feature_dim,self.feature_dim)
        self.out = nn.Linear(self.feature_dim,2)
        self.dropout = nn.Dropout(dropout)
        self.init_params(self,initialize)

    def forward(self,X):
        feature = self.activations(X)
        predicted = self.out(self.dropout(feature))

        return predicted


    def activations(self,X):
        emb = self.embeddings(X).unsqueeze(1) # inserts a dimension of size in given dimension

        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]

        pools = [F.max_pool1d(conv,conv.size(2)).squeeze(2) for conv in convs]

        predictions = torch.cat(pools,1)

        fc = self.fc(predictions)

        predictions = torch.sigmoid(fc)*F.relu(fc)
        predictions +=(1. - torch.sigmoid(fc))*predictions

        return predictions

    def init_params(self,initialize):
        for param in self.parameters():
            if param.requires_grad and len(param.shape)>0:
                stddev = 1/math.sqrt(param.shape[0])
            if initialize == 'uniform':
                torch.nn.init.uniform_(param,a=-0.05,b=0.05)
            else:
                torch.nn.init.normal_(param,stddev=stddev)

