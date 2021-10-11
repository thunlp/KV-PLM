from transformers import BertTokenizer
import pickle
import random
import numpy as np
import pdb
tokenizer0 = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')


import sys
filename = sys.argv[-1]
f = open('BC5CDR/'+filename+'.txt', 'r')
lines = f.readlines()[2:]
f.close()

mark2id = {'O':0, 'B-Entity':1, 'I-Entity':2, 'X':-1}
alltok = []
allatt = []
allmsk = []
alllab = []

tok = [102]
att = [1]
msk = [0]
lab = [-1]
for line in lines:
    wd = line.strip('\n').split('\t')
    if len(wd)<3:
        tok += [103]
        att += [1]
        msk += [0]
        lab += [-1]
        if len(tok)>20:#70
            #if len(tok)>128:
            #    pdb.set_trace()
            tmptok = np.zeros(128)
            tmpatt = np.zeros(128)
            tmpmsk = np.zeros(128)
            tmplab = np.ones(128)*-1
            tmptok[:min(128, len(tok))] = tok[:min(128, len(tok))]
            tmpatt[:min(128, len(att))] = att[:min(128, len(att))]
            tmpmsk[:min(128, len(msk))] = msk[:min(128, len(msk))]
            tmplab[:min(128, len(lab))] = lab[:min(128, len(lab))]
            alltok.append(tmptok)
            allatt.append(tmpatt)
            allmsk.append(tmpmsk)
            alllab.append(tmplab)
            tok = [102]
            att = [1]
            msk = [0]
            lab = [-1]
    else:
        word = wd[0]
        mark = mark2id[wd[-1]]
        enc = tokenizer0.encode(word)[1:-1]
        tok += enc
        att += [1]*len(enc)
        msk += [1]
        lab += [mark]
        if len(enc)>1:
            msk += [0]*(len(enc)-1)
            lab += [-1]*(len(enc)-1)
        
tmptok = np.zeros(128)
tmpatt = np.zeros(128)
tmpmsk = np.zeros(128)
tmplab = np.ones(128)*(-1)
tmptok[:min(128, len(tok))] = tok[:min(128, len(tok))]
tmpatt[:min(128, len(att))] = att[:min(128, len(att))]
tmpmsk[:min(128, len(msk))] = msk[:min(128, len(msk))]
tmplab[:min(128, len(lab))] = lab[:min(128, len(lab))]
alltok.append(tmptok)
allatt.append(tmpatt)
allmsk.append(tmpmsk)
alllab.append(tmplab)
print(len(alllab), alllab[1])
np.save(filename+'_tok.npy', alltok)
np.save(filename+'_lab.npy', alllab)
np.save(filename+'_att.npy', allatt)
np.save(filename+'_msk.npy', allmsk)
