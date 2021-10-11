from transformers import BertTokenizer
#from tokenization import SmilesTokenizer
import pickle
import random
import numpy as np

tokenizer0 = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
'''
tokenizer0.vocab.pop('[unused1]')
tokenizer0.vocab.pop('[unused2]')
tokenizer0.vocab.pop('[unused3]')
tokenizer0.vocab.pop('[unused4]')
tokenizer0.vocab['<<']=1
tokenizer0.vocab['>>']=2
tokenizer0.vocab['[[']=3
tokenizer0.vocab[']]']=4

special_tokens_dict = {'additional_special_tokens': ['<<','>>','[[',']]']}
tokenizer0.add_special_tokens(special_tokens_dict)
'''

import sys
import json

filename = sys.argv[-1]
f = open(filename+'.txt')
corpus = f.readlines()
f.close()
labels = pickle.load(open('name_labels.pkl', 'rb'))
alltok = []
alllab = []
allatt = []
for index in range(len(corpus)):
    if index%1000==0:
        print(index)
    js = json.loads(corpus[index].strip('\n'))
    sent = js['text']
    if js['label'] not in labels:
        labels[js['label']] = len(labels)
    lab = labels[js['label']]
    tokens = tokenizer0(sent, padding=True, truncation=True, max_length=128)
    tok = np.zeros(128)
    att = np.zeros(128)
    tok[:len(tokens['input_ids'])] = tokens['input_ids']
    att[:len(tokens['attention_mask'])] = tokens['attention_mask']
    alltok.append(tok.astype('int64'))
    alllab.append(lab)
    allatt.append(att.astype('int64'))
np.save('predata/'+filename+'_tok.npy', alltok)
np.save('predata/'+filename+'_lab.npy', alllab)
np.save('predata/'+filename+'_att.npy', allatt)

