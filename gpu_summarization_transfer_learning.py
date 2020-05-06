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

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
cached_lemmatize = lru_cache(maxsize=50000)(WordNetLemmatizer().lemmatize)
from gensim.utils import simple_preprocess, to_unicode

STOP_WORDS = ["i", "a", "about", "an", "are", "as", "at", "be", "by", 
                "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", 
                "this", "to", "was", "what", "when", "where", "who", "will", "with"]

def ExpandContractions(contraction):

    contraction = re.sub(r"won\'t", "will not", contraction)
    contraction = re.sub(r"can\'t", "can not", contraction)

    contraction = re.sub(r"n\'t", " not", contraction)
    contraction = re.sub(r"\'re", " are", contraction)
    contraction = re.sub(r"\'s", " is", contraction)
    contraction = re.sub(r"\'d", " would", contraction)
    contraction = re.sub(r"\'ll", " will", contraction)
    contraction = re.sub(r"\'t", " not", contraction)
    contraction = re.sub(r"\'ve", " have", contraction)
    contraction = re.sub(r"\'m", " am", contraction)

    return contraction

def PreProcess(line):
    
    line = line.translate(str.maketrans("", "", string.punctuation))
    line = ExpandContractions(line)
    line = simple_preprocess(to_unicode(line))
    line = [cached_lemmatize(word) for word in line if word not in STOP_WORDS]

    line = " ".join(line)
    return line

class LineSentenceGenerator(object):

    def __init__(self, source, preprocess=None, max_sentence_length=10000, limit=None, preprocess_flag=True):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.input_files = []

        if preprocess != None and callable(preprocess) and preprocess_flag:
            self.preprocess = preprocess
        else:
            self.preprocess = lambda line: line.rstrip("\r\n")

        if isinstance(self.source, list):
#             print('List of files given as source. Verifying entries and using.')
            self.input_files = [filename for filename in self.source if os.path.isfile(filename)]
            self.input_files.sort()  # makes sure it happens in filename order

        elif os.path.isfile(self.source):
#             print('Single file given as source, rather than a list of files. Wrapping in list.')
            self.input_files = [self.source]  # force code compatibility with list of files

        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
#             print('Directory of files given as source. Reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('Input is neither a file nor a path nor a list')
#         print('Files read into LineSentenceGenerator: %s' % ('\n'.join(self.input_files)))

        self.token_count = 0

    def __iter__(self):
        for file_name in self.input_files:
#             print('Reading file %s', file_name)
            with open(file_name, 'rb') as fin:
                for line in itertools.islice(fin, self.limit):
                    line = self.preprocess(utils.to_unicode(line))
                    self.token_count += len(line)
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length

    def __len__(self):
        if self.token_count > 0:
            return self.token_count
        else:
            return len(self.input_files)

    def __bool__(self):
        return self.has_data()

    def is_empty(self):
        return len(self.input_files) == 0

    def has_data(self):
        return not self.is_empty()

    
def chunk_data(data, n):
    for i in range(0, len(data), n):
        yield data[i:i+n]#.text, data[i:i+n].summ
        


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


# %%
def generate_summaries(data, model, tokenizer, outfile, outfile_true, device="cpu", max_length=150, min_length=50, batch_size=128, start_token=None):
    
    
    with open(outfile, "w", encoding="utf-8") as predictions, open(outfile_true, "w", encoding="utf-8") as gold_standard:
        for batch_data in tqdm(chunk_data(data, batch_size)):
            model.to(device)
            batch_text = [d.text for d in batch_data]
            batch_summary = [d.summ for d in batch_data]
            
            inputs = tokenizer.batch_encode_plus(batch_text, max_length=1024, return_tensors='pt', pad_to_max_length=True)

            summaries = model.generate(input_ids=inputs['input_ids'].to(device), 
                                    attention_mask=inputs["attention_mask"].to(device), 
                                    max_length=max_length + 2,  
                                    min_length=min_length + 1, 
                                    num_beams=5, 
                                    no_repeat_ngram_size=3,
                                    early_stopping=True,
                                    decoder_start_token_id=start_token)

            outputs = [tokenizer.decode(summary, skip_special_tokens=True, clean_up_tokenization_spaces=False) for summary in summaries]
            
            for summ, true in zip(outputs, batch_summary):
                predictions.write(summ.rstrip("\r\n") + "\n")
                predictions.flush()
                gold_standard.write(true.rstrip("\r\n") + "\n")
                gold_standard.flush()
    
            model.cpu()


def generate_summaries_no_chunk(data, model, tokenizer, outfile, outfile_true, device="cpu", max_length=150, min_length=50, batch_size=128, start_token=None):
    
    with open(outfile, "w", encoding="utf-8") as predictions, open(outfile_true, "w", encoding="utf-8") as gold_standard:

        batch_text = [d.text for d in data]
        batch_summary = [d.summ for d in data]

        model.to(device)
        
        inputs = tokenizer.batch_encode_plus(batch_text, max_length=1024, return_tensors='pt', pad_to_max_length=True)

        summaries = model.generate(input_ids=inputs['input_ids'].to(device), 
                                attention_mask=inputs["attention_mask"].to(device), 
                                max_length=max_length + 2,  
                                min_length=min_length + 1, 
                                num_beams=5, 
                                no_repeat_ngram_size=3,
                                early_stopping=True,
                                decoder_start_token_id=start_token)

        outputs = [tokenizer.decode(summary, skip_special_tokens=True, clean_up_tokenization_spaces=False) for summary in summaries]
        
        for summ, true in zip(outputs, batch_summary):
            predictions.write(summ.rstrip("\r\n") + "\n")
            predictions.flush()
            gold_standard.write(true.rstrip("\r\n") + "\n")
            gold_standard.flush()

        model.cpu()

# %%
from torchtext.data import Dataset,Example
from torchtext.data import Field, BucketIterator

SRC = Field(sequential = False,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = False)

TRG = Field(sequential = False,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = False)

def read_data(X, y, preprocess=None, limit=1000):
    examples = []
    fields = {'text-tokens': ('text', SRC),
              'summ-tokens': ('summ', TRG)}
    for i,(x,y) in enumerate(zip(LineSentenceGenerator(X,preprocess),LineSentenceGenerator(y,preprocess))):
        if i > limit:
            break
        text_field = x
        summ_field = y
       
        e = Example.fromdict({"text-tokens": text_field, "summ-tokens": summ_field},
                             fields=fields)
        examples.append(e)
    print("examples: \n", examples[0])
    return Dataset(examples, fields=[('text', SRC), ('summ', TRG)])


# %%
train_data = read_data(train_file_X, train_file_y, preprocess=None, limit=1000)
test_data = read_data(test_file_X, test_file_y, preprocess=None, limit=200)
val_data = read_data(val_file_X, val_file_y, preprocess=None, limit=200)

SRC.build_vocab(train_data.text, min_freq = 2,max_size=20000)
TRG.build_vocab(train_data.summ, min_freq = 2)


# %%
BATCH_SIZE = 16
# BATCH_SIZE = 64
MAX_LENGTH = 300
MIN_LENGTH = 50


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
t5_type = "t5-small"
bart_type = "bart-large-cnn"


# %%
results_t5 = os.path.join(cnn_dailymail_path, "t5.prediction")
results_bart = os.path.join(cnn_dailymail_path, "bart.prediction")

labels_t5 = os.path.join(cnn_dailymail_path, "t5.true")
labels_bart = os.path.join(cnn_dailymail_path, "bart.true")


# %%
t5_model = T5ForConditionalGeneration.from_pretrained(t5_type)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_type)

parameters = t5_model.config.task_specific_params
if parameters is not None:
    t5_model.config.update(parameters.get("summarization", {}))


# %%
generate_summaries_no_chunk(train_data, t5_model, t5_tokenizer, results_t5, labels_t5, device, MAX_LENGTH, MIN_LENGTH, BATCH_SIZE, None)


# %%
bart_model = BartForConditionalGeneration.from_pretrained(bart_type)
bart_tokenizer = BartTokenizer.from_pretrained(bart_type)


# %%
generate_summaries_no_chunk(train_data, bart_model, bart_tokenizer, results_bart, labels_bart, device, MAX_LENGTH, MIN_LENGTH, BATCH_SIZE, bart_model.config.eos_token_id)


# %%



# %%


