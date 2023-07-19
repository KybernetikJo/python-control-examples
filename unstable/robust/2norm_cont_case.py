import numpy as np
import scipy.linalg as linalg

import control as ct
print(ct.__version__)

import slycot
print(slycot.__version__)

A1 = np.array([[0.0, 1.0],[-0.5, -0.1]])
B1 = np.array([[0.],[1.]])
C1 = np.eye(2)
D1 = np.zeros((2,1))

sys = ct.ss(A1, B1, C1, D1)

print(ct.h2norm(sys))
print(ct.h2norm(sys))

print(ct.norm(sys))
print(ct.norm(sys,p=2))