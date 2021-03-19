import numpy as np

def read_data_give_chunks(floc):
    f = open(floc,'r')
    lines = f.readlines()
    data = []
    for l in lines:
        data.append(float(l[:-1]))
    f.close()
    chunks = [(data[x], data[x - 3:x][::-1]) for x in range(3, len(data))]
    return chunks

def cal_mse(chunks):
    running_error = 0.0
    for t in range(len(chunks)):
        dotpdt = np.dot(weights.reshape(1,3),np.array(chunks[t][1]).reshape((3,1)))
        diffsq = np.square(chunks[t][0] - dotpdt)
        running_error += diffsq
    return running_error/len(chunks)

f0_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw4/hw4_nasdaq00.txt'
f1_loc = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw4/hw4_nasdaq01.txt'
chunks0 = read_data_give_chunks(f0_loc)
chunks1 = read_data_give_chunks(f1_loc)

A = np.zeros((3,3))
b = np.zeros((3,1))
for t in range(len(chunks0)):
    mat1 = np.array(chunks0[t][1]).reshape((3,1))
    mat2 = mat1.view().reshape((1,3))
    A = np.add(A,mat1 @ mat2)
    b = np.add(b,chunks0[t][0] * mat1)

weights = (np.linalg.inv(A)) @ b
print('a1 = ', weights[0][0])
print('a2 = ', weights[1][0])
print('a3 = ', weights[2][0])
print('MSE for 2000: ', cal_mse(chunks0)[0][0])
print('MSE for 2001: ', cal_mse(chunks1)[0][0])