
f = open('choice_items_sci.txt')
items = f.readlines()
f.close()

f = open('PCdes_choice_ID2.txt')
IDs = f.readlines()
f.close()

f = open('PCdes_choice_sm2.txt')
sms = f.readlines()
f.close()

import numpy as np

seq = np.arange(1428)
np.random.shuffle(seq)

f1 = open('PCdes_for_human0725.txt', 'w')
f2 = open('PCdes_human_ans0725.txt', 'w')
st = ['A', 'B', 'C', 'D']

for i in range(200):
    sq = seq[i]
    f1.write('【'+str(i)+'】'+'\n')
    f1.write('PubChem ID: '+IDs[sq]+'\n')
    f1.write('SMILES: '+sms[sq]+'\n')
    chs = np.arange(4)
    np.random.shuffle(chs)
    for j in range(4):
        if chs[j]==0:
            f2.write(st[j]+'\n')
        f1.write(st[j]+'. '+items[sq*4+chs[j]])
    f1.write('\n')

f1.close()
f2.close()

