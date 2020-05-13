import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from utils import *
import math
import os


def word2idx_mapper(text,vocab_dict):
    """ 
    Preprocess -> Tokenize and map the word tokens to vocab indexes
    : return list of indexes of the words.
    """
    text = '<sos> ' + text + ' <eos>'
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(text)
#     print(tokens)
    ids = [vocab_dict[i] for i in tokens]
    return ids


def idx2seq(idx_list,vocab_list):
    """
    : param idx_list : list of indices
    : param vocab_list : index2word mapping list
    : return summary sequence
    Converts the list of indexes back to sequence of words
    """
    return " ".join(vocab_list[i] for i in idx_list)

def generate_summary(text,model,vocab_list,vocab_dict,max_len=150,device='cpu'):
    """
    :param text       : actual text
    :param model      : transformer model
    :param vocab_list : idx2word list of the vocab
    :param vocab_dict : word2idx dict of the vocab
    :param max_len    : max length of the generated text, 150 by default
    :param device     : device to store the tensors in, cpu by default
    """
    src = torch.Tensor(word2idx_mapper(text,vocab_dict)).long().unsqueeze(1).to(device)
    
    memory = model.transformer.encoder(model.pos_enc(model.embed_src(src) * math.sqrt(model.d_model)))
    out_idx = [vocab_dict['<sos>'],]
    
    for i in range(max_len):
        trg = torch.LongTensor(out_idx).unsqueeze(1).to(device)
        
        output = model.fc(model.transformer.decoder(model.pos_enc(model.embed_tgt(trg) * math.sqrt(model.d_model)), memory))

        out_token = output.argmax(2)[-1].item()
        
        out_idx.append(out_token)
        
        if out_token == vocab_dict['<eos>']:
            break
        
        
    return idx2seq(out_idx,vocab_list)

if __name__ == "__main__":
    print("Cannot run standalone, please use it as import generate_summary\n")
