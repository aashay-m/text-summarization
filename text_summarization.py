import os
import re
import string
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import torch.optim as optim
import itertools

from tqdm.notebook import tqdm

import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors

from torchtext.data import Dataset,Example
from torchtext.data import Field, BucketIterator
from torchtext.vocab import FastText

from einops import rearrange

from utils import *
from data import get_src_trg, read_data
from model import TransformerSummarizer
from trainers import train, evaluate

BATCH_SIZE = 128
SEQ_LEN = 4000

TRAIN_SIZE = 50000
TEST_SIZE = 5000
VAL_SIZE = 5000

D_MODEL = 300 # Embedding dimension
DIM_FEEDFORWARD = 300  # Dimensionality of the hidden state

ATTENTION_HEADS = 6  # number of attention heads
N_LAYERS = 1 # number of encoder/decoder layers

N_EPOCHS = 1
CLIP = 1

device = torch.device('cpu')

base_dir = "cnn_dm"
train_file_X = os.path.join(base_dir,"train.source")
train_file_y = os.path.join(base_dir,"train.target")
test_file_X = os.path.join(base_dir,"test.source")
test_file_y = os.path.join(base_dir,"test.target")
val_file_X = os.path.join(base_dir,"val.source")
val_file_y = os.path.join(base_dir,"val.target")

out_dir = os.path.join(os.getcwd(), "results", "transformer")


if __name__ == '__main__':

    SRC, TRG = get_src_trg(True)
    VOCAB_SIZE = len(SRC.vocab)  

    train_data = read_data(train_file_X, train_file_y, SRC, SRC, PreProcessMinimal, TRAIN_SIZE)
    test_data = read_data(test_file_X, test_file_y, SRC, SRC, PreProcessMinimal, TEST_SIZE)
    val_data = read_data(val_file_X, val_file_y, SRC, SRC, PreProcessMinimal, VAL_SIZE)

    SRC.build_vocab(train_data.text, min_freq = 2)

    src_list = SRC.vocab.itos  # index2word
    src_dict = SRC.vocab.stoi # word2index

    PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

    train_iter = BucketIterator(train_data, BATCH_SIZE, shuffle=True, sort_key=lambda x: len(x.text), sort_within_batch=True)
    val_iter = BucketIterator(val_data, BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)
    test_iter = BucketIterator(test_data, BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)

    ff = FastText("en")
    embeddings =  ff.get_vecs_by_tokens(SRC.vocab.itos)

    model = TransformerSummarizer(ATTENTION_HEADS, N_LAYERS, N_LAYERS, DIM_FEEDFORWARD, \
                                    SEQ_LEN, VOCAB_SIZE, PAD_IDX, embeddings=embeddings).to(device)

    parameters = filter(lambda p:p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_iter, num_batches,optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_iter,val_batches, criterion, "evaluate")

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        
    test_size = math.ceil(len(test_data)/BATCH_SIZE)
    test_loss = evaluate(model, test_iter, test_size, criterion, "testing")

    print(f'| Test Loss: {test_loss:.3f}')

    print(f'Saving Model')
    torch.save(model.state_dict(), os.path.join(out_dir, "transformer_model.pt"))


    with open(os.path.join(out_dir, "raw.txt"), "w", encoding="utf-8") as text, \
            open(os.path.join(out_dir, "pred.txt"), "w", encoding="utf-8") as pred, \
                open(os.path.join(out_dir, "true.txt"), "w", encoding="utf-8") as true:

        for iter, batch in enumerate(test_iter):

            src = batch.text
            trg = batch.summ
            trg_inp, trg_out = trg[:-1, :], trg[1:, :]

            output = model(src, trg)
            output = F.softmax(output, dim=-1)
            output = output_test.argmax(-1)

            raw_text = " ".join([src_list[i] for i in src.squeeze(1).transpose(0,1)[0].tolist()])
            true_summary = " ".join([src_list[i] for i in trg.squeeze(1).transpose(0,1)[0].tolist()])
            prediction = " ".join([src_list[i] for i in output.transpose(0,1)[0].tolist()])

            # print(output.transpose(0,1)[0].shape)
            # print("text: ", raw_text)
            # print("\n\nsumm: ", true_summary)
            # print("\n\npred: ", prediction)

            text.write(raw_text + "\n")
            true.write(true_summary + "\n")
            pred.write(prediction + "\n")
