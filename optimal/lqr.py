# lqg.py
# Johannes Kaisinger, July 2023
#
# Demonstrate a lqg regulator
#
# [1] 

import control as ct
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

c1 = 1.;
c2 = 1.;
d1 = 0.1;
d2 = 0.1;
m = 1.;


A = np.array([[0, 1],[-(c1+c2)/m, -(d1+d2)/m]])
B = np.array([[0], [1/m]])
C = np.eye(2)
D = np.zeros((2,1))

sys = ct.ss(A,B,C,D,name='plant')
sys

Q = np.eye(2)
R = np.eye(1)

Kc, _, _ = ct.lqr(A,B,Q,R)
ctrl, clsys = ct.create_statefbk_iosystem(sys, Kc)
print(type(ctrl))
print(type(clsys))