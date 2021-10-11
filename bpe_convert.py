import numpy as np
import sys
pth = sys.argv[1]
out = sys.argv[2]
ifact = sys.argv[3]

f = open('bpe_vocab.txt')
lines = f.readlines()
f.close()
sub ={}

for line in lines:
    wd = line.strip('\n').split(' ')
    sub[wd[0]] = len(sub)

f = open(pth)
lines = f.readlines()
f.close()
alltrain = []
'''
import pickle
sup = pickle.load(open('sup_dic.pkl', 'rb'))
'''
if ifact=='1':
    for i in range(int(len(lines)/2)):
        line = lines[i*2].strip('\n').split(' ')
        tmp = []
        for wd in line:
            if wd in sub:
                tmp.append(sub[wd])
            else:
                #if wd not in sup:
                tmp.append(len(sub))
                #    sup[wd] = -1-len(sup)
                #tmp.append(sup[wd])
        tmp+=[362]
        line = lines[i*2+1].strip('\n').split(' ')
        for wd in line:
            if wd in sub:
                tmp.append(sub[wd])
            else:
                #if wd not in sup:
                tmp.append(len(sub))
                #    sup[wd] = -1-len(sup)
                #tmp.append(sup[wd])
        alltrain.append(tmp)
else:
    for i in range(len(lines)):
        line = lines[i].strip('\n').split(' ')
        tmp = []
        for wd in line:
            if wd in sub:
                tmp.append(sub[wd])
            else:
                #if wd not in sup:
                tmp.append(len(sub))
                #    sup[wd] = -1-len(sup)
                #tmp.append(sup[wd])
        alltrain.append(tmp)

#pickle.dump(sup, open('sup_dic.pkl', 'wb'))
np.save(out, alltrain)
