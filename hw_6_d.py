import math
import numpy as np
from prettytable import PrettyTable

def read_data(floc,y=0):
    f = open(floc,'r')
    lines = f.readlines()
    data = []
    for l in lines:
        l = l.split(' ')
        if not y:
            l = l[:-1]
            d = [int(x) for x in l]
        else:
            l[-1] = l[-1][:-1]
            d = int(l[0])
        data.append(d)
    return data

def pdct(xt):
    return np.product(np.power(1-pi_a, xt))

def cal_ll():
    t_sum = 0.0
    t_mis = 0
    for xt,yt in zip(d_x, d_y):
        if yt:
            py_x = 1-pdct(xt)
            t_sum += math.log(py_x)
            if py_x<=0.5:
                t_mis+=1
        else:
            py_x = pdct(xt)
            t_sum += math.log(py_x)
            if py_x<0.5:
                t_mis+=1
    return t_sum/267.0, t_mis


def cal_zi_xi_xy(xt,yt):
    xt_a= np.asarray(xt)
    xt_a *= yt
    return xt_a * pi_a / (1-pdct(xt))

fn_x = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw6/noisyOr_X.txt'
fn_y = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw6/noisyOr_Y.txt'

d_x = read_data(fn_x)
d_x[-1].append(1)
d_y = read_data(fn_y,y=1)

pi = [1/23.0 for _ in range(23)]
pi_a = np.asarray(pi)
ll=[]
mistakes=[]
data_array = np.asarray(d_x)
s = data_array.sum(axis = 0)

for i in range(257):
    ll.append(cal_ll())
    p_zi_xi_xy = np.zeros([23])
    for xt,yt in zip(d_x, d_y):
        temp = cal_zi_xi_xy(xt, yt)
        p_zi_xi_xy += temp
    pi_a = np.asarray(p_zi_xi_xy/s)

ind = [0,1,2,4,8,16,32,64,128,256]
t = PrettyTable(['Iteration','M', 'LL'])
for i in ind:
    t.add_row([i,ll[i][1],ll[i][0]])
print(t)