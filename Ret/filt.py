import pickle

filt = pickle.load(open('../../dataset/PubChem/CP-filt.pkl', 'rb'))
synid = pickle.load(open('../align/synid_dic.pkl', 'rb'))

f = open('../align/align_des_all.txt')
corpus = f.readlines()
f.close()

cor = {}
f= open('align_des_filt3.txt', 'w')
cnt = 0
for i in range(len(synid)):
    tag = False
    cid = synid[i]
    line = corpus[i]
    namelist = filt[cid]
    namelist.sort(key = lambda i:len(i),reverse=True)
    for nm in namelist:
        alin = line.lower()
        anm = nm.lower()
        if len(anm)>3 and alin.find(anm)>-1:
            tag = True
            tmp = line[alin.find(anm):alin.find(anm)+len(nm)]
            line = line.replace(tmp, 'it')#[MASK]
        if line.find(nm)>-1:
            tag=True
        line = line.replace(nm, 'it')
    #if tag:
    cnt+=1
    f.write(line)
    cor[len(cor)] = i
#pickle.dump(cor, open('cor_id.pkl', 'wb'))
