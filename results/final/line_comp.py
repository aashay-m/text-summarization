import os
from tqdm import tqdm
base_dir = os.path.dirname(os.path.abspath(__file__))


mismatch_list = []
with open(os.path.join(base_dir,"t5.prediction"),encoding="utf-8") as pred, open(os.path.join(base_dir,"t5.true"),encoding="utf-8") as true: 
    for i, (x, y) in tqdm(enumerate(zip(pred, true))):
        x_set = set(x.strip().split())
        y_set = set(y.strip().split())
        if len(x_set.intersection(y_set))==0:
            mismatch_list.append(i)
        else:
            with open(os.path.join(base_dir,"t5_corrected.true"),"a+",encoding="utf-8") as f:
                f.write(y)
            with open(os.path.join(base_dir,"t5_corrected.pred"),"a+",encoding="utf-8") as f:
                f.write(x)
            


print(mismatch_list)
print(len(mismatch_list))
        