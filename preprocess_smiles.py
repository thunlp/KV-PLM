import sys

pth = sys.argv[1]
out = sys.argv[2]
ifact = sys.argv[3]
print(ifact)
f = open(pth)
lines = f.readlines()
f.close()
f = open(out, 'w')

for line in lines:
    newline = line#.upper()
    if ifact=='1':
        wds = newline.strip('\n').split('>>')
    else:
        wds = [newline.strip('\n')]
    for wd in wds:
        sent1 = wd.upper()
        sent0 = wd#-1
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
        f.write(sent+'\n')
        
f.close()
