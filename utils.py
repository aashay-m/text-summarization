import math
from functools import lru_cache
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
import gensim.utils as utils
import itertools
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
cached_lemmatize = lru_cache(maxsize=50000)(WordNetLemmatizer().lemmatize)
from gensim.utils import simple_preprocess, to_unicode

import re
import string

import os


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

    def __init__(self, source, preprocess=None, max_sentence_length=4000, limit=None, preprocess_flag=True):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.input_files = []

        if preprocess != None and callable(preprocess) and preprocess_flag:
            self.preprocess = preprocess
        else:
            self.preprocess = lambda line: line.rstrip("\r\n")

        if isinstance(self.source, list):
            # print('List of files given as source. Verifying entries and using.')
            self.input_files = [filename for filename in self.source if os.path.isfile(filename)]
            self.input_files.sort()  # makes sure it happens in filename order

        elif os.path.isfile(self.source):
            # print('Single file given as source, rather than a list of files. Wrapping in list.')
            self.input_files = [self.source]  # force code compatibility with list of files

        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            # print('Directory of files given as source. Reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('Input is neither a file nor a path nor a list')
        # print('Files read into LineSentenceGenerator: %s' % ('\n'.join(self.input_files)))

        self.token_count = 0

    def __iter__(self):
        for file_name in self.input_files:
            print('Reading file %s'%(file_name))
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