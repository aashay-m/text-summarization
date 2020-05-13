import math

from torchtext.data import Dataset,Example
from torchtext.data import Field, BucketIterator
from torchtext.data.utils import get_tokenizer

from utils import *

def get_src_trg(tokenize=True):

    if tokenize == False:
        SRC = Field(sequential = False,
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = False)

        TRG = Field(sequential = False,
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = False)
    else:
        SRC = Field(tokenize = get_tokenizer("spacy"),
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = False)

        TRG = Field(tokenize = get_tokenizer("spacy"),
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = False)
    
    return SRC, TRG


def read_data(X, y, SRC, TRG, preprocess=None, limit=1000):

    examples = []
    fields = {'text-tokens': ('text', SRC),
              'summ-tokens': ('summ', TRG)}

    for i,(x,y) in enumerate(zip(LineSentenceGenerator(X, preprocess),LineSentenceGenerator(y, preprocess))):
        text_field = x
        summ_field = y

        if limit is not None and i > limit:
            break

        e = Example.fromdict({"text-tokens": text_field, "summ-tokens": summ_field}, fields=fields)
        examples.append(e)

    return Dataset(examples, fields=[('text', SRC), ('summ', TRG)])
