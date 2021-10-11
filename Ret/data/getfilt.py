

f = open('align_des_sent3.txt', 'r')
lines = f.readlines()
f.close()
f = open('filt_sent.txt', 'w')

lines = [line.strip('\n').strip(' ') for line in lines]
lines.sort()

lst = lines[0]
i = 1
while i<len(lines):
    samecnt = 0
    while lines[i]==lst:
        samecnt += 1
        i += 1
    if samecnt>=4:
        f.write(lst+'\n')
    lst = lines[i]
    i += 1

f.close()
