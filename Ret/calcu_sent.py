import torch
import random
from math import log
import numpy as np
import pickle
#cor = np.load('predata_sent/test_cor.npy')
test_ans = np.load('sent/test_ans.npy', allow_pickle=True)
dessyn = torch.from_numpy(np.load('output_sent/filt_des.npy'))
dessmi = torch.from_numpy(np.load('output_sent/filt_des.npy'))
syn = torch.from_numpy(np.load('output_sent/test_smi.npy'))
smi = torch.from_numpy(np.load('output_sent/test_smi.npy'))

f = open('data/filt_sent.txt', 'r')
lines = f.readlines()
f.close()
f = open('data/choice_items_sci1.txt', 'w')

alldic = {}
sumsmi = 0
sumsyn = 0
cnt = 0
for i in range(3000):
    ans = test_ans[i]
    if len(ans)==0:
        continue
    alltmp = []
    ans_ind = ans[random.randint(0,len(ans)-1)]
    alltmp.append(ans_ind)
    smi_score = torch.cosine_similarity(smi[i],dessmi[ans_ind], dim=-1)
    syn_score = torch.cosine_similarity(syn[i],dessyn[ans_ind], dim=-1)
    f.write(lines[ans_ind])
    tagsmi = True
    tagsyn = True
    for j in range(3):
        #ind = random.randint(ans_ind+10,ans_ind+860)%870
        confli = True
        while confli:
            ind = random.randint(0,869)
            tmpcnt = False
            for an in ans:
                if abs(an-ind)<10:
                    tmpcnt = True
                    break
            if not tmpcnt:
                confli = False
        alltmp.append(ind)
        if torch.cosine_similarity(smi[i],dessmi[ind], dim=-1)>smi_score:
            tagsmi = False
        if torch.cosine_similarity(syn[i],dessyn[ind], dim=-1)>syn_score:
            tagsyn = False
        f.write(lines[ind])
    alldic[i] = alltmp
    if tagsmi:
        sumsmi += 1
    if tagsyn:
        sumsyn += 1
    cnt += 1

print(sumsmi/cnt, sumsyn/cnt)
print(cnt)
