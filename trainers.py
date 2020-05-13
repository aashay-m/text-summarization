from tqdm import tqdm
from utils import chunk_data

import torch
device = torch.device('cpu')

def train(model: nn.Module, iterator: BucketIterator, num_batches: int, optimizer: optim.Optimizer, criterion: nn.Module, clip: float):
    
    print("Training Started")
    epoch_loss = 0
    model.train()

    for batch in tqdm(iterator,total=num_batches):

        src = batch.text
        trg = batch.summ
        trg_inp, trg_out = trg[:-1, :], trg[1:, :]

        optimizer.zero_grad()

        output = model(src.to(device), trg_inp.to(device))
        output = output.view(-1, output.shape[-1])

        loss = criterion(output, trg_out.view(-1))
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        
    print("Training Done")

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, iterator: BucketIterator, num_batches:int, criterion: nn.Module, desc: str):

    print(f'{desc}ing')
    epoch_loss = 0
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(iterator,total=num_batches):
            
            src = batch.text
            trg = batch.summ
            trg_inp, trg_out = trg[:-1, :], trg[1:, :]

            output = model(src.to(device), trg_inp.to(device))
            output = output.view(-1,output.shape[-1])

            loss = criterion(output, trg_out.view(-1))
            epoch_loss += loss.item()
            
    print(f"{desc}ing Done")

    return epoch_loss / len(iterator)


def generate_summaries(data, model, tokenizer, outfile, outfile_true, max_length=150, min_length=50, batch_size=128, start_token=None, mode="w"):
    
    with open(outfile, mode, encoding="utf-8") as predictions, open(outfile_true, mode, encoding="utf-8") as gold_standard:
        for batch_data in tqdm(chunk_data(data, batch_size), total=len(data)):
            # model.to(device)
            batch_text = [d.text for d in batch_data]
            batch_summary = [d.summ for d in batch_data]
            
            inputs = tokenizer.batch_encode_plus(batch_text, max_length=1024, return_tensors='pt', pad_to_max_length=True)

            summaries = model.generate(input_ids=inputs['input_ids'],#.to(device), 
                                    attention_mask=inputs["attention_mask"],#.to(device), 
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
    
            # model.cpu()

def generate_summaries_no_chunk(data, model, tokenizer, outfile, outfile_true, max_length=150, min_length=50, batch_size=128, start_token=None, mode="w"):
    
    with open(outfile, mode, encoding="utf-8") as predictions, open(outfile_true, mode, encoding="utf-8") as gold_standard:

        batch_text = [d.text for d in data]
        batch_summary = [d.summ for d in data]

        # model.to(device)
        
        inputs = tokenizer.batch_encode_plus(batch_text, max_length=1024, return_tensors='pt', pad_to_max_length=True)

        summaries = model.generate(input_ids=inputs['input_ids'],#.to(device), 
                                attention_mask=inputs["attention_mask"],#.to(device), 
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

        # model.cpu()