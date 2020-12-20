import numpy as np
from numpy.linalg import inv

x = [[1,3],[1,6],[1,7],[1,8],[1,11]]
print(x)
y = [13,8,11,2,6]
print(y)

xtxi = inv(np.matmul(np.transpose(x),x))
print(xtxi)
xtxixt = np.matmul(xtxi,np.transpose(x))
print(xtxixt)
coefficients = np.matmul(xtxixt,y)
print(coefficients)