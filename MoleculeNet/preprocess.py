
f = open('../text_sider.txt', 'r')
lines = f.readlines()
fw = open('smiles_sider.txt', 'w')

label = []

for line in lines:
    line = line.strip('\n')
    sent1 = line.upper()
    sent0 = line#-1
    sent = ''
    tag = False
    for pos in range(len(sent0)):
        if sent1[pos]!=sent0[pos] and (sent0[pos]=='c' or sent0[pos]=='n' or sent0[pos]=='o'):
            if tag:
                sent += '='
                tag = False
            else:
                tag = True
        if sent0[pos]=='c' or sent0[pos]=='n' or sent0[pos]=='o':
            sent+=sent1[pos]
        else:
            sent += sent0[pos]

    sent = sent.replace('(', '')
    sent = sent.replace(')', '')
    for i in range(10):
        item = str(i)
        sent = sent.replace(item, '')
    fw.write(sent+'\n')
  
f.close()
fw.close()

