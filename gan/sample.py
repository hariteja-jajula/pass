import os
import argparse
import torch
from model import NetG
from utils import load_dataset
import sys

import re
def special_match(strg, search=re.compile(r'[^a-z0-9.~`!@#$%^&*()-_+=|\\{}[\]\'";:?/<>, ]').search):
    return not bool(search(strg))

parse = argparse.ArgumentParser()
parse.add_argument("--dataset", default="Dataset/password.txt", type=str)
parse.add_argument("--savepath", default="sample/gen_password.txt", type=str)
parse.add_argument("--modelpath", default="gan/pretrained/passwordG_model.pth", type=str)
parse.add_argument("--numsample", default=1000000, type=int)
parse.add_argument("--gpu", default=True, type=bool)
parse.add_argument("--batchsize", default=1000, type=int)
parse.add_argument("--length", default=16, type=int)
args = parse.parse_args()

path = args.dataset
save_path = args.savepath
model_path = args.modelpath
num_sample = args.numsample
batch_size = args.batchsize
seq_len = args.length
use_gpu = args.gpu
current_size = 0

train_set, text, train_len, vocab_len = load_dataset(
    root=path,
    batch_size=batch_size,
    seq_len=seq_len)

with open(save_path, "w+") as f:
    while current_size < num_sample:
        inputs = torch.randn(batch_size, 128).float()
        model = torch.load(model_path).eval()
        if use_gpu:
            inputs = inputs.cuda()
            model = model.cuda()
        
        output = model(inputs)
        sample = output.argmax(2).cpu()   
        i=0
        k=0    
        for i in range(batch_size):
            gen_pass = ""
            for j in range(seq_len):   
                gen_pass += text.vocab.itos[sample[i][j]]
            gen_pass = str(gen_pass).replace("'", "") 
            if re.search('[a-zA-Z]', gen_pass) and special_match(gen_pass) and len(gen_pass)>=4 and len(gen_pass)<=16:
                f.write(gen_pass+ "\n")
                current_size+=1
            if current_size==1000000:
                sys.exit()
            

                  
    
    f.close()
    