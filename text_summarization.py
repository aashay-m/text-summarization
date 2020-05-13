import os
import re
import string
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import torch.optim as optim
import itertools

from tqdm import tqdm
from pathlib import Path

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
from generate_summary import generate_summary

MAX_LENGTH = 250
SEQ_LEN = 4000

D_MODEL = 300 # Embedding dimension
DIM_FEEDFORWARD = 300  # Dimensionality of the hidden state

ATTENTION_HEADS = 6  # number of attention heads
N_LAYERS = 3 # number of encoder/decoder layers

CLIP = 1

device = torch.device('cpu')

# base_dir = os.path.join(os.getcwd(), "data")
base_dir = os.path.join(os.getcwd(), "cnn_dm")

train_file_X = os.path.join(base_dir,"train.source")
train_file_y = os.path.join(base_dir,"train.target")
test_file_X = os.path.join(base_dir,"test.source")
test_file_y = os.path.join(base_dir,"test.target")
val_file_X = os.path.join(base_dir,"val.source")
val_file_y = os.path.join(base_dir,"val.target")

out_dir = os.path.join(os.getcwd(), "results", "transformer", "full")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Text Summaries using a Transformer Network')

    parser.add_argument('--training_size', type=int, help='Size of the training set to use (0 for the full set)', default=0)
    parser.add_argument('--test_size', type=int, help='Size of the test set to use (0 for the full set)', default=5000)

    parser.add_argument('--epochs', type=int, help='Number of Epochs to use for training', default=25)
    parser.add_argument('--batch_size', type=int, help='Batch Size to use for training', default=128)

    parser.add_argument('--evaluate_only', action='store_true', help='Simply load and evaluate model',default=False)

    parser.add_argument('--model_path', type=str, help='Test a pretrained transformer model', default="")
    parser.add_argument('--output_path', type=str, help='Directory where results should be written', default="")

    args = parser.parse_args()

    if args.output_path != "":
        outdir = args.output_path

    TRAIN_SIZE = None if args.training_size == 0 else args.training_size
    TEST_SIZE = None if args.test_size == 0 else args.test_size
    VAL_SIZE = 5000
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    SRC, TRG = get_src_trg(True)

    train_data = read_data(train_file_X, train_file_y, SRC, SRC, PreProcessMinimal, TRAIN_SIZE)
    val_data = read_data(val_file_X, val_file_y, SRC, SRC, PreProcessMinimal, VAL_SIZE)
    test_data = read_data(test_file_X, test_file_y, SRC, SRC, PreProcessMinimal, TEST_SIZE)

    SRC.build_vocab(train_data, test_data, val_data, min_freq=10)
    VOCAB_SIZE = len(SRC.vocab)

    print("Data read. Vocab Size %s" % VOCAB_SIZE)

    src_list = SRC.vocab.itos  # index2word
    src_dict = SRC.vocab.stoi # word2index

    PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

    train_iter = BucketIterator(train_data, BATCH_SIZE, shuffle=True, sort_key=lambda x: len(x.text), sort_within_batch=True)
    val_iter = BucketIterator(val_data, BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)
    test_iter = BucketIterator(test_data, BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)

    if not args.evaluate_only:
            
        ff = FastText("en")
        embeddings =  ff.get_vecs_by_tokens(SRC.vocab.itos)

        model = TransformerSummarizer(ATTENTION_HEADS, N_LAYERS, N_LAYERS, DIM_FEEDFORWARD, \
                                        SEQ_LEN, VOCAB_SIZE, PAD_IDX, embeddings=embeddings).to(device)

        num_batches = math.ceil(len(train_data)/BATCH_SIZE)
        val_batches = math.ceil(len(val_data)/BATCH_SIZE)

        parameters = filter(lambda p:p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        print("Training Started")

        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss = train(model, train_iter, num_batches, optimizer, criterion)
            valid_loss = evaluate(model, val_iter, val_batches, criterion, "evaluat")

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')
            torch.save(model.state_dict(), os.path.join(out_dir, "transformer_model.pt"))
            
        test_size = math.ceil(len(test_data)/BATCH_SIZE)
        test_loss = evaluate(model, test_iter, test_size, criterion, "test")

        print(f'| Test Loss: {test_loss:.3f}')

        print("Training Done")
        print("Saving Model")
        torch.save(model.state_dict(), os.path.join(out_dir, "transformer_model.pt"))

    else:
        if args.model_path == "":
            raise FileNotFoundError

        model = TransformerSummarizer(ATTENTION_HEADS, N_LAYERS, N_LAYERS, DIM_FEEDFORWARD, \
                                        SEQ_LEN, VOCAB_SIZE, PAD_IDX).to(device)
        model.load_state_dict(args.model_path)

    Path.mkdir(out_dir, parents=True, exist_ok=True)

    with open(os.path.join(out_dir, "raw.txt"), "w", encoding="utf-8") as text, \
            open(os.path.join(out_dir, "pred.txt"), "w", encoding="utf-8") as pred, \
                open(os.path.join(out_dir, "true.txt"), "w", encoding="utf-8") as true:

        for data in tqdm(test_data, total=TEST_SIZE):

            src_text = data.text
            trg_text = data.summ

            raw_text = " ".join(src_text)
            true_summary = " ".join(trg_text)
            prediction = generate_summary(raw_text, model, src_list, src_dict, MAX_LENGTH)

            text.write(raw_text + "\n")
            true.write(true_summary + "\n")
            pred.write(prediction + "\n")