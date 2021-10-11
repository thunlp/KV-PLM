from transformers import BertTokenizer
from tokenization import SmilesTokenizer
import pickle
import random
import numpy as np
import sys
tokenizer0 = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

f = open('align_des_filt3.txt')
corpus = f.readlines()
f.close()

fn = sys.argv[-1]
#cor = pickle.load(open('predata_sent/cor.pkl', 'rb'))
sm = pickle.load(open('data/align_sub_all.pkl', 'rb'))
#smsyn = pickle.load(open('../../dataset/PubChem/CP-name-all.pkl', 'rb'))
smsyn = pickle.load(open('data/align_smdic.pkl', 'rb'))
synid = pickle.load(open('data/synid_dic.pkl', 'rb'))
import pdb

alltokdes = []
allattdes = []
alltoksmi = []
allattsmi = []
alltoksyn = []
allattsyn = []
rec_cor = []
cnt = 0

if fn=='train':
    a=0
    b=10500
elif fn=='dev':
    a=10500
    b=12000
else:
    a=12000
    b=15000
for ind in range(a, b):
    index = ind#seq[ind]
    sent = corpus[index].strip('\n')
    
    sub = [102] + [i+30700 for i in sm[index][:30]] + [103]
    
    if synid[index] not in smsyn:
        smsyn[synid[index]] = 'unknown'
    subsyn = tokenizer0(smsyn[synid[index]], padding=True, truncation=True, max_length=32)['input_ids']
    
    toksmi = np.zeros(32)
    attsmi = np.zeros(32)
    toksmi[:min(32, len(sub))] = sub
    attsmi[:min(32, len(sub))] = np.ones(min(32, len(sub)))

    toksyn = np.zeros(64)
    attsyn = np.zeros(64)
    toksyn[:min(64, len(subsyn))] = subsyn
    attsyn[:min(64, len(subsyn))] = np.ones(min(64, len(subsyn)))

    alltoksyn.append(toksyn.astype('int64'))
    allattsyn.append(attsyn.astype('int64'))
    alltoksmi.append(toksmi.astype('int64'))
    allattsmi.append(attsmi.astype('int64'))
    rec_cor.append(cnt)
    for st in sent.split('.'):
        if len(st.split(' '))<5:
            continue
        tokens = tokenizer0(st, padding=True, truncation=True, max_length=128)['input_ids']
        tokdes = np.zeros(64)
        attdes = np.zeros(64)
        tokdes[:min(64, len(tokens))] = tokens[:min(64, len(tokens))]
        attdes[:min(64, len(tokens))] = np.ones(min(64, len(tokens)))
        cnt+=1
        alltokdes.append(tokdes.astype('int64'))
        allattdes.append(attdes.astype('int64'))
    '''
    tokens = tokenizer0(sent, padding=True, truncation=True, max_length=64)['input_ids']
    tokdes = np.zeros(64)
    attdes = np.zeros(64)
    tokdes[:min(64, len(tokens))] = tokens[:min(64, len(tokens))]
    attdes[:min(64, len(tokens))] = np.ones(min(64, len(tokens)))
    alltokdes.append(tokdes.astype('int64'))
    allattdes.append(attdes.astype('int64'))
    '''
rec_cor.append(cnt)
#print(cnt)
np.save('sent/'+fn+'_cor.npy', rec_cor)
np.save('sent/'+fn+'_tokdes.npy', alltokdes)
np.save('sent/'+fn+'_attdes.npy', allattdes)
np.save('sent/'+fn+'_toksmi.npy', alltoksmi)
np.save('sent/'+fn+'_attsmi.npy', allattsmi)
np.save('sent/'+fn+'_toksyn.npy', alltoksyn)
np.save('sent/'+fn+'_attsyn.npy', allattsyn)
