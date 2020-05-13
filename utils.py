import math
import re
import string
import os
import itertools

import gensim
from gensim.models import KeyedVectors
import gensim.utils as utils
import gensim.downloader as api

from tqdm import tqdm
from functools import lru_cache

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

def PreProcessMinimal(line):

    line = line.translate(str.maketrans("", "", string.punctuation))
    line = ExpandContractions(line)
    line = [word.lower() for word in line.split()]

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
            self.input_files = [filename for filename in self.source if os.path.isfile(filename)]
            self.input_files.sort()  # makes sure it happens in filename order

        elif os.path.isfile(self.source):
            self.input_files = [self.source]  # force code compatibility with list of files

        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('Input is neither a file nor a path nor a list')

        self.token_count = 0

    def __iter__(self):
        for file_name in self.input_files:
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
        yield data[i:i+n]
        

def get_remaining_data(data, true_file):

    final_data = []
    already_done = open(true_file, "r", encoding="utf-8").readlines()
    already_done = [sent.rstrip("\r\n") for sent in already_done]

    for example in data:
        if example.summ in already_done:
            continue
        final_data.append(example)
    
    print("Pruned data to process from %s to %s" % (len(data.examples), len(final_data)))
    data.examples = final_data

    return data


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
