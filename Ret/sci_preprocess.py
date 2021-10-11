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


f = open('align_smiles.txt')
smiles = f.readlines()
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
    
       
    sms = smiles[index].strip('\n')
    toksms = tokenizer0(sms, padding=True, truncation=True, max_length=32)['input_ids']
    toksmi = np.zeros(32)
    attsmi = np.zeros(32)
    toksmi[:min(32, len(toksms))] = toksms[:min(32, len(toksms))]
    attsmi[:min(32, len(toksms))] = np.ones(min(32, len(toksms)))


    alltoksmi.append(toksmi.astype('int64'))
    allattsmi.append(attsmi.astype('int64'))
    rec_cor.append(cnt)
    for st in sent.split('.'):
        if len(st.split(' '))<5:
            continue
        tokens = tokenizer0(st, padding=True, truncation=True, max_length=128)['input_ids']
        tokdes = np.zeros(32)
        attdes = np.zeros(32)
        tokdes[:min(32, len(tokens))] = tokens[:min(32, len(tokens))]
        attdes[:min(32, len(tokens))] = np.ones(min(32, len(tokens)))
        cnt+=1
        alltokdes.append(tokdes.astype('int64'))
        allattdes.append(attdes.astype('int64'))
    
rec_cor.append(cnt)
#print(cnt)
np.save('sci/'+fn+'_cor.npy', rec_cor)
np.save('sci/'+fn+'_tokdes.npy', alltokdes)
np.save('sci/'+fn+'_attdes.npy', allattdes)
np.save('sci/'+fn+'_toksmi.npy', alltoksmi)
np.save('sci/'+fn+'_attsmi.npy', allattsmi)

