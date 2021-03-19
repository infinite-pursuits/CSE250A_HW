import math
from matplotlib import pyplot as plt

def update_val(x):
    sum = 0.0
    for k in range(10):
        k+=1
        sum += math.tanh(x+(1/k))
    return sum/10.0

def val(x):
    sum = 0.0
    for k in range(10):
        k+=1
        sum += math.log(math.cosh(x+(1/k)))
    return sum/10.0

x_init = -2
x_list = [x_init]
x_n = x_init
for i in range(20):
    x_np1 = x_n - update_val(x_n)
    x_list.append(x_np1)
    x_n = x_np1

print(x_list)
print(val(x_list[-1]))
plt.plot(x_list)
plt.xlabel('iteration #')
plt.ylabel('updated x')
plt.show()
plt.savefig('x vs Iteration')