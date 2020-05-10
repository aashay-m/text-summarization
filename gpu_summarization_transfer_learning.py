# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
import string 
import os
import math
import itertools
import sys

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

# BATCH_SIZE = 16
BATCH_SIZE = 4
MAX_LENGTH = 250
MIN_LENGTH = 50

# %%
# pretrained_embeddings = api.load("fasttext-wiki-news-subwords-300")


# %%
# !wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_dm.tgz 
# !tar -xzvf cnn_dm.tgz


# %%
cnn_dailymail_path = os.path.join(os.getcwd(), "cnn_dm/")

train_file_X = os.path.join(cnn_dailymail_path, "train.source")
train_file_y = os.path.join(cnn_dailymail_path, "train.target")
test_file_X = os.path.join(cnn_dailymail_path, "test.source")
test_file_y = os.path.join(cnn_dailymail_path, "test.target")
val_file_X = os.path.join(cnn_dailymail_path, "val.source")
val_file_y = os.path.join(cnn_dailymail_path, "val.target")

t5_type = "t5-small"
bart_type = "bart-large-cnn"

cnn_dailymail_out_path = os.path.join(cnn_dailymail_path, "output")
t5_results = os.path.join(cnn_dailymail_out_path, "t5.prediction")
bart_results = os.path.join(cnn_dailymail_out_path, "bart.prediction")

t5_labels = os.path.join(cnn_dailymail_out_path, "t5.true")
bart_labels = os.path.join(cnn_dailymail_out_path, "bart.true")

# %%

# %%
SRC, TRG = get_src_trg(tokenize=False)

train_data = read_data(train_file_X, train_file_y, SRC, TRG, preprocess=None, limit=1000)
test_data = read_data(test_file_X, test_file_y, SRC, TRG, preprocess=None, limit=200)
val_data = read_data(val_file_X, val_file_y, SRC, TRG, preprocess=None, limit=200)

SRC.build_vocab(train_data.text, min_freq = 2,max_size=20000)
TRG.build_vocab(train_data.summ, min_freq = 2)


# %%



# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# %%
bart_model = BartForConditionalGeneration.from_pretrained(bart_type)
bart_tokenizer = BartTokenizer.from_pretrained(bart_type)

generate_summaries(train_data, bart_model, bart_tokenizer, bart_results, bart_labels, device, MAX_LENGTH, MIN_LENGTH, BATCH_SIZE, bart_model.config.eos_token_id)
