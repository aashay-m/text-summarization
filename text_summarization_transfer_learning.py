import re
import string 
import os
import math
import itertools
import sys
import argparse

from tqdm import tqdm
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Transformer

# !{sys.executable} -m pip install transformers

from transformers import T5ForConditionalGeneration, T5Tokenizer   
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

import gensim
import gensim.utils as utils
import gensim.downloader as api
from gensim.models import KeyedVectors

from utils import *
from data import *
from trainers import generate_summaries, generate_summaries_no_chunk

# pretrained_embeddings = api.load("fasttext-wiki-news-subwords-300")

# !wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_dm.tgz 
# !tar -xzvf cnn_dm.tgz


# BATCH_SIZE = 16
BATCH_SIZE = 4
MAX_LENGTH = 250
MIN_LENGTH = 50


cnn_dailymail_path = os.path.join(os.getcwd(), "cnn_dm/")
cnn_dailymail_out_path = os.path.join(cnn_dailymail_path, "output")

train_file_X = os.path.join(cnn_dailymail_path, "train.source")
train_file_y = os.path.join(cnn_dailymail_path, "train.target")
test_file_X = os.path.join(cnn_dailymail_path, "test.source")
test_file_y = os.path.join(cnn_dailymail_path, "test.target")
val_file_X = os.path.join(cnn_dailymail_path, "val.source")
val_file_y = os.path.join(cnn_dailymail_path, "val.target")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Summarize Text Using Pretrained Models')

    parser.add_argument('--method', required=True, choices=["t5", "bart"], type=str, help='Pretrained model to use')
    parser.add_argument('--restart', action='store_true', help='Whether to start afresh or resume',default=False)

    args = parser.parse_args()

    
    SRC, TRG = get_src_trg(tokenize=False)

    train_data = read_data(train_file_X, train_file_y, SRC, TRG, preprocess=None)
    test_data = read_data(test_file_X, test_file_y, SRC, TRG, preprocess=None)
    val_data = read_data(val_file_X, val_file_y, SRC, TRG, preprocess=None)

    SRC.build_vocab(train_data.text, min_freq=2, max_size=20000)
    TRG.build_vocab(train_data.summ, min_freq=2)

    if args.method == "t5":

        results_file = os.path.join(cnn_dailymail_out_path, "t5.prediction")
        labels_file = os.path.join(cnn_dailymail_out_path, "t5.true")

        model_type = "t5-small"
        model = T5ForConditionalGeneration.from_pretrained(model_type)
        tokenizer = T5Tokenizer.from_pretrained(model_type)
        parameters = model.config.task_specific_params
        if parameters is not None:
            model.config.update(parameters.get("summarization", {}))

        decoder_start_token_id = None


    elif args.method == "bart":

        results_file = os.path.join(cnn_dailymail_out_path, "bart.prediction")
        labels_file = os.path.join(cnn_dailymail_out_path, "bart.true")

        model_type = "bart-large-cnn"
        model = BartForConditionalGeneration.from_pretrained(model_type)
        tokenizer = BartTokenizer.from_pretrained(model_type)

        decoder_start_token_id = model.config.eos_token_id

    if args.restart == True:
        mode = "w"
    else:
        mode = "a"
        train_data = get_remaining_data(train_data, labels_file)

    generate_summaries(train_data, model, tokenizer, results_file, labels_file, MAX_LENGTH, MIN_LENGTH, BATCH_SIZE, decoder_start_token_id, mode=mode)




