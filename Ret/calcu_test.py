import pdb
import torch

import numpy as np
cor = np.load('sent/test_cor.npy')
dessyn = torch.from_numpy(np.load('output_sent/test_des.npy'))
#dessmi = torch.from_numpy(np.load('output0/test_des8065.npy'))
syn = torch.from_numpy(np.load('output_sent/test_smi.npy'))
#smi = torch.from_numpy(np.load('output0/test_smi8065.npy'))
'''
pdb.set_trace()
score_smi = torch.zeros(3000,3000)
for i in range(3000):
    score = torch.cosine_similarity(smi[i], dessmi, dim=-1)
    for j in range(3000):
        score_smi[i,j] = sum(score[cor[j]:cor[j+1]])/(cor[j+1]-cor[j])

rec_smi = []
for i in range(3000):
    a,idx = torch.sort(score_smi[i])
    for j in range(3000):
        if idx[-1-j]==i:
            rec_smi.append(j)
            break

print(sum( (np.array(rec_smi)<1).astype(int) ))
print(sum( (np.array(rec_smi)<5).astype(int) ))
print(sum( (np.array(rec_smi)<10).astype(int) ))
print(sum( (np.array(rec_smi)<20).astype(int) ))

pdb.set_trace()
'''
score_syn = torch.zeros(3000,3000)
for i in range(3000):
    score = torch.cosine_similarity(syn[i], dessyn, dim=-1)
    for j in range(3000):
        score_syn[i,j] = sum(score[cor[j]:cor[j+1]])/(cor[j+1]-cor[j])

rec_syn = []
'''
for i in range(3000):
    a,idx = torch.sort(score_syn[i])
    for j in range(3000):
        if idx[-1-j]==i:
            rec_syn.append(j)
            break
'''
for i in range(3000):
    a,idx = torch.sort(score_syn[:,i])
    for j in range(3000):
        if idx[-1-j]==i:
            rec_syn.append(j)
            break
print(sum( (np.array(rec_syn)<1).astype(int) ))
print(sum( (np.array(rec_syn)<5).astype(int) ))
print(sum( (np.array(rec_syn)<10).astype(int) ))
print(sum( (np.array(rec_syn)<20).astype(int) ))
