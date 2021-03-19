import random
import matplotlib.pyplot as plt

def sample():
    t = [random.choice([0, 1]) for _ in range(10)]
    return t

def fb(s):
    sum = 0.0
    for ind,val in enumerate(s):
        sum += pow(2,ind)*val
    return sum

prev_p = []
num = 0.0
deno = 0.0
i = 1 #1/3/5/7/9
for _ in range(10000000):
    s = sample()
    pz_bs = 0.666667 * pow(0.2, abs(128.0 - fb(s)))
    num += (pz_bs * s[i])
    deno += pz_bs
    if not deno:
        continue
    pq_e = num/deno
    prev_p.append(pq_e)

plt.plot(prev_p)
plt.xlabel('sample #')
plt.ylabel('Probability i = '+str(i+1))
plt.show()
plt.savefig('prob'+str(i+1))