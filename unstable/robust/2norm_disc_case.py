import numpy as np
import scipy.linalg as linalg

import control as ct
print(ct.__version__)

import slycot
print(slycot.__version__)

sys = ct.tf([1,-2.841,2.875,-1.004],[1,-2.417,2.003,-0.5488],0.1);

A1 = np.array([[0.5, 0.0],[0.0, 0.625]])
B1 = np.array([[0.],[1.]])
C1 = np.eye(2)
D1 = np.zeros((2,1))


print(ct.h2norm(sys))
print(ct.h2norm(sys))

print(ct.norm(sys))
print(ct.norm(sys,p=2))