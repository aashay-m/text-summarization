import math
from torchtext.data import Dataset,Example
from torchtext.data import Field, BucketIterator
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
        SRC = Field(tokenize = "spacy",
                lower = False)

        TRG = Field(tokenize = "spacy",
                is_target = True,
                lower = False)
    
    return SRC, TRG


# def prepare_data(train_file_paths,test_file_paths,val_file_paths,debug_flag=True):

#     train_data = read_data(train_file_paths[0],train_file_paths[1],PreProcess,1000)
#     test_data = read_data(test_file_paths[0],test_file_paths[1],PreProcess,200)
#     val_data = read_data(val_file_paths[0],val_file_paths[1],PreProcess,200)

#     if debug_flag:
#         debug(train_data)

#     return (train_data,test_data,val_data)



# def debug(data):
#     print(data.fields)
#     print(data[0].text)
#     print(data[0].summ)



def read_data(X, y, SRC, TRG, preprocess=None, limit=1000):
    examples = []
    fields = {'text-tokens': ('text', SRC),
              'summ-tokens': ('summ', TRG)}
    for i,(x,y) in enumerate(zip(LineSentenceGenerator(X, preprocess),LineSentenceGenerator(y, preprocess))):
        text_field = x
        summ_field = y
       
        e = Example.fromdict({"text-tokens": text_field, "summ-tokens": summ_field},
                             fields=fields)
        examples.append(e)
    print("examples: \n", examples[0])
    return Dataset(examples, fields=[('text', SRC), ('summ', TRG)])
