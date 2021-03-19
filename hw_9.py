import numpy as np
from prettytable import PrettyTable

def read_files(floc):
    f = open(floc, 'r')
    lines = f.readlines()
    data = np.zeros([81,81])
    for l in lines:
        l = l.strip().split()
        data[int(l[0])-1,int(l[1])-1] = float(l[2])
    f.close()
    return data

def read_rewards(floc):
    f = open(floc, 'r')
    lines = f.readlines()
    data = np.zeros([81])
    for i,l in enumerate(lines):
        l = l.strip().split()
        data[i] = int(l[0])
    f.close()
    return data

def value_iteration():
    pl = [p_a1, p_a2, p_a3, p_a4]
    for i in range(1000):
        for s in range(81):
            m_vals = []
            for p in pl:
                s_val = 0.0
                for sd in range(81):
                    s_val += p[s][sd] * v_pi[sd]
                m_vals.append(gamma * s_val)
            v_pi[s] = max(rw[s] + m_vals)
    return v_pi

def cal_vp(p):
    temp = np.zeros([81,81])
    d = {0:p_a1, 1:p_a2, 2:p_a3, 3:p_a4}
    for i in range(81):
        for j in range(81):
            pol = d[int(p[i])]
            temp[i][j] = -gamma * pol[i][j]
        temp[i][i] += 1
    inv = np.linalg.inv(temp)
    vs = np.dot(inv, rw.reshape([81, 1]))
    return vs

def policy_iteration():
    p = np.ones([81])
    vp_old = np.zeros([81])
    v = np.ones([81])
    pl = [p_a1, p_a2, p_a3, p_a4]
    for i in range(10):
        if not np.allclose(vp_old,v,atol=0.01):
            vp_old = v
            v = cal_vp(p)
            for s in range(81):
                m_vals = []
                for p_l in pl:
                    s_val = 0.0
                    for sd in range(81):
                        s_val += p_l[s][sd] * v[sd]
                    m_vals.append(s_val[0])
                p[s] = np.argmax([m_vals])
    return v, p


p_a1l = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw9/prob_a1.txt'
p_a2l= '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw9/prob_a2.txt'
p_a3l = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw9/prob_a3.txt'
p_a4l = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw9/prob_a4.txt'
rwl = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw9/rewards.txt'

p_a1 = read_files(p_a1l)
p_a2 = read_files(p_a2l)
p_a3 = read_files(p_a3l)
p_a4 = read_files(p_a4l)
rw = read_rewards(rwl)
v_pi = np.zeros([81])
gamma = 0.99
v1 = value_iteration()
v,p = policy_iteration()


d = {0:u'\u2190',1:u'\u2191',2:u'\u2192',3:u'\u2193'}

t = PrettyTable(['state','V','action'])
for i in range(81):
    t.add_row([i+1,v[i][0],d[p[i]]])
print(t)

