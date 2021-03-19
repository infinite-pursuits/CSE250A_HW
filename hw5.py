import numpy as np
import math
from matplotlib import pyplot as plt

def read_data(floc,y=0):
    f = open(floc,'r')
    lines = f.readlines()
    data = []
    for l in lines:
        l = l.split(' ')
        l = l[:-1]
        d = [int(x) for x in l]
        d.append(y)
        data.append(d)
    return data

def sig(x):
    return 1 / (1 + math.exp(-x))

def accuracy(dataset):
    correct = 0
    total = len(dataset)
    for xt in dataset:
        yt = xt[-1]
        xt = xt[:-1]
        xt = np.asarray(xt)
        dot = np.dot(w, xt)
        sd = sig(dot)
        if sd >= 0.5:
            y_pred = 1
        else:
            y_pred = 0
        if y_pred == yt:
            correct += 1
    return float(correct)/total

f3train_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw5/train3.txt'
f5train_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw5/train5.txt'
f3test_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw5/test3.txt'
f5test_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw5/test5.txt'
x_train3 = read_data(f3train_loc)
x_train5 = read_data(f5train_loc,y=1)
x_test3 = read_data(f3test_loc)
x_test5 = read_data(f5test_loc,y=1)
#3-y=0, 5-y=1, last element of each datapoint is its label

x_train = x_train3+x_train5

w = np.zeros((64))
t_lw_list =[]
for _ in range(10): #change to 20k for gradient
    t_lw = 0.0
    grad_l = np.zeros((64))
    hessian = np.zeros((64, 64))
    for xt in x_train:
        yt = xt[-1]
        xt = xt[:-1]
        xt = np.asarray(xt)
        dot = np.dot(w, xt)
        sd = sig(dot)
        smd = sig(-dot)
        lw = (yt * math.log(sd)) + ((1 - yt) * math.log(smd))

        t_lw += lw
        grad_l += (yt - sd) * xt

        xxt = np.matmul(xt.reshape([64, 1]), xt.reshape([1, 64]))
        hessian += (-1 * sd * smd * xxt)

    hess_inv = np.linalg.inv(hessian)
    w -= np.matmul(hess_inv , grad_l)
    #w += ((0.2/1400)*grad_l)
    print(t_lw)
    t_lw_list.append(t_lw)

print(1-accuracy(x_train3), 1-accuracy(x_train5), 1- accuracy(x_test3), 1- accuracy(x_test5))
print(w.reshape(8,8))
plt.plot(t_lw_list)
plt.xlabel('iteration #')
plt.ylabel('Log Likelihood')
plt.show()
plt.savefig('LL vs Iteration')