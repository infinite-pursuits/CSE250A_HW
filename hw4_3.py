import numpy as np
from prettytable import PrettyTable
import math
from matplotlib import pyplot as plt

def read_data(floc, iflag=0, biflag=0):
    f = open(floc,'r')
    lines = f.readlines()
    data = []
    dict = {}
    for l in lines:
        d = l[:-1]
        if not biflag:
            if iflag:
                d = int(d)
        else:
            split_line = d.split('\t')
            i1, i2, c = int(split_line[0]), int(split_line[1]), int(split_line[2])
            d = (i1, i2, c)
            dict[(i1,i2)] = c
        data.append(d)
    f.close()
    return data, dict

def compute_lu_lb(sent):
    f = 1
    p_u = 1.0
    p_b = 1.0
    for ind, w_d in enumerate(sent):
        p_u *= p_u_w_quick[w_d]
        if ind:
            w = sent[ind-1]
        else:
            w = '<s>'
        p_b *= p_b_wd_w_dict[w][w_d]
        if not p_b_wd_w_dict[w][w_d]:
            f = 0
            print(w, w_d)
    lu = math.log(p_u)
    if f:
        lb = math.log(p_b)
    else:
        lb = 'INDET'
    return lu, p_u, lb, p_b

fv_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw4/hw4_vocab.txt'
fu_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw4/hw4_unigram.txt'
fb_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw4/hw4_bigram.txt'

vocab, _ = read_data(fv_loc)
u_counts, _ = read_data(fu_loc, iflag=1)
bi_counts, bi_dict = read_data(fb_loc, biflag=1)
assert len(vocab) == len(u_counts)

t_u_counts = sum(u_counts)
p_u_w = np.divide(u_counts, t_u_counts)
p_u_w_quick = {}
t = PrettyTable(['Token', 'p_w_unigram'])
for i, w in enumerate(vocab):
    p_u_w_quick[w] = p_u_w[i]
    if w[0] == 'A':
        t.add_row([w, p_u_w[i]])

print(t)

c_b_mat = []
c_b_w = {}
for k,v in bi_dict.items():
    w1 = k[0]
    if w1 in c_b_w:
        c_b_w[w1] += v
    else:
        c_b_w[w1] = v


p_b_wd_w_dict = {}
for i in range(len(vocab)):
    for j in range(len(vocab)):
        if (i+1, j+1) in bi_dict:
            temp = bi_dict[(i+1, j+1)]/c_b_w[i+1]
        else:
            temp = 0.0
        if vocab[i] in p_b_wd_w_dict:
            p_b_wd_w_dict[vocab[i]][vocab[j]] = temp
        else:
            p_b_wd_w_dict[vocab[i]] = {vocab[j]: temp}

top_5_the = sorted(p_b_wd_w_dict['THE'].items(), key=lambda x: x[1], reverse=True)[:5]
t2 = PrettyTable(['Token', 'p_bigram'])
for i in range(5):
    t2.add_row(top_5_the[i])
print(t2)

sent = ['LAST','WEEK','THE','STOCK','MARKET','FELL','BY','ONE','HUNDRED','POINTS']
t3 = PrettyTable(['Model', 'MLL'])
umll, _, bimll, _ = compute_lu_lb(sent)
t3.add_row(['UNIGRAM',umll])
t3.add_row(['BIGRAM',bimll])
print(t3)

sent2 = ['THE', 'NINETEEN', 'OFFICIALS', 'SOLD', 'FIRE', 'INSURANCE']
t4 = PrettyTable(['Model', 'MLL'])
umll, _, bimll, _ = compute_lu_lb(sent2)
t4.add_row(['UNIGRAM',umll])
t4.add_row(['BIGRAM',bimll])
print(t4)

p_m = 1.0
l_lamb = {}

for lamb in list(np.arange(0,1,0.01)):
    p_m = 1.0
    for ind,w_d in enumerate(sent2):
        if ind:
            w = sent2[ind-1]
        else:
            w = '<s>'
        p_m_w_wd = ((1 - lamb) * p_u_w_quick[w_d]) + (lamb*p_b_wd_w_dict[w][w_d])
        p_m *= p_m_w_wd
    l_lamb[lamb] = math.log(p_m)

plt.plot(l_lamb.keys(),l_lamb.values())
plt.xlabel('lambda')
plt.ylabel('log likelihoods')
plt.show()
plt.savefig('LL vs lambda')

print('MAXIMIZING LAMBDA IS : ',max(l_lamb, key=l_lamb.get), 'MAXIMUM LL IS: ', l_lamb[max(l_lamb, key=l_lamb.get)])