import math
import torch
import torch.nn as nn
import torch.optim as optim
from data import prepare_data,SRC,TRG
from model import TransformerSummarizer
from torchtext.data import Field, BucketIterator
from torchtext.vocab import FastText
import os

import time
from tqdm import tqdm

base_dir = "data"
train_files = [os.path.join(base_dir,"train.source"),os.path.join(base_dir,"train.target")]
test_files = [os.path.join(base_dir,"test.source"),os.path.join(base_dir,"test.target")]
val_files = [os.path.join(base_dir,"val.source"),os.path.join(base_dir,"val.target")]

train_data,test_data,val_data = prepare_data(train_files,test_files,val_files)

SRC.build_vocab(train_data.text, min_freq = 2,max_size=4000)
TRG.build_vocab(train_data.summ, min_freq = 2)

SEQ_LEN = 4000

D_MODEL = 200 #embedding_size
DIM_FEEDFORWARD = 200
VOCAB_SIZE = len(SRC.vocab)
print(VOCAB_SIZE)
ATTENTION_HEADS = 6 # emb_dim must be divisible by number of heads (300 in our case)
N_LAYERS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

def make_iters(train_data,val_data,test_data):

    train_iter = BucketIterator(train_data,BATCH_SIZE, shuffle=True,
                                                    sort_key=lambda x: len(x.text), sort_within_batch=True)

    val_iter = BucketIterator(val_data, BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)
    test_iter = BucketIterator(test_data,BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)

    return train_iter,val_iter,test_iter

def load_pretrained(field):
    ff = FastText("en")
    embeddings =  ff.get_vecs_by_tokens(field.vocab.itos)
    return embeddings

train_iter,val_iter,test_iter = make_iters(train_data,val_data,test_data)

embeddings = load_pretrained(SRC)

model = TransformerSummarizer( ATTENTION_HEADS,N_LAYERS, N_LAYERS, DIM_FEEDFORWARD, SEQ_LEN,VOCAB_SIZE,embeddings=embeddings).to(device)

PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)



def train(model: nn.Module,
          iterator: BucketIterator,
          num_batches: int,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    
    print("Training......")

    model.train()

    epoch_loss = 0

    for batch in tqdm(iterator,total=num_batches):
        
#         if i == 1:
#             break

        src = batch.text
        trg = batch.summ
        
#         tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
#         tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to('cuda')

#         trg_inp = trg[:,:-1] 

        optimizer.zero_grad()

        output = model(src.to(device), trg.to(device))

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        
    print("Training Done.....")

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             num_batches:int,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0
    
    print("Evaluating....")
    with torch.no_grad():

        for batch in tqdm(iterator,total=num_batches):
            
#             if i == 1:
#                 break
            src = batch.text
            trg = batch.summ

            output = model(src.to(device), trg.to(device))

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            
        print("Evaluating Done........")

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

parameters = filter(lambda p: p.requires_grad,model.parameters())
optimizer = optim.Adam(parameters)

num_batches = math.ceil(len(train_data)/BATCH_SIZE)
val_batches = math.ceil(len(val_data)/BATCH_SIZE)

N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, num_batches,optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_iter,val_batches, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
test_size = math.ceil(len(test_data)/BATCH_SIZE)
test_loss = evaluate(model, test_iter,test_size, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


