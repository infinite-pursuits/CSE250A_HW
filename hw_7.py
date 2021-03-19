import numpy as np
from matplotlib import pyplot as plt

def read_f(loc, fflag=0):
    data = []
    with open(loc) as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(' ')
            if fflag==1:
                l = line.split('\t')
                l[-1] = l[-1][:-1]
                d = [float(x) for x in l]
            elif fflag==0:
                l = l[:-1]
                d = [int(x) for x in l]
            elif fflag==2:
                l = l[:-1]
                d = [float(x) for x in l]
            data.append(d)
    return data

obs = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw7/observations.txt' #fflag=0
ems = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw7/emissionMatrix.txt' #fflag=1
trans = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw7/transitionMatrix.txt' #fflag=2

ist = [0.037037037037 for x in range(27)]
o = np.array(read_f(obs,fflag=0)[0])
b = np.array(read_f(ems, fflag=1))
a = np.array(read_f(trans, fflag=2))
pi = np.array(ist)
n = 27
T = 175000

l = np.zeros([n,T])
l[:,0] = np.log(pi)+np.log(b[:, o[0]])

for tp1 in range(1,T):
    t = tp1-1
    #there is no j here unlike the formula coz we consider all js. we also consider all
    #i's while taking max. hence basically all of A matrix.
    temp = np.tile(l[:,t],(27,1))+np.log(a)
    l[:,tp1] = np.max(temp,axis=1) + np.log(b[:,o[tp1]])

f_states = np.zeros([T], dtype='int')
f_states[T-1] = np.argmax(l[:,T-1])

for t in range(T-2, -1, -1):
    f_states[t] = np.argmax(l[:,t] + np.log(a[:,f_states[t+1]]))

print('FSTATES',f_states)
f_states+=1
plt.plot(f_states)
plt.xlabel('time')
plt.ylabel('states')
plt.show()
plt.savefig('states vs time')

sent = [chr(ord('a')+int(f_states[0])-1)]
print('sent',sent)
for i in range(1,T):
    if f_states[i]!=f_states[i-1]:
        if f_states[i]==27:
            print(' ')
        else:
            print(chr(ord('a')+int(f_states[i])-1))