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
C = np.array([[1, 0]])
D = np.array([[0]])

sys = ct.ss(A,B,C,D,name='P')
sys

Q = np.eye(2)
R = np.eye(1)

Qe = np.eye(2)
Re = np.eye(1)

G = np.eye(2)

Kc, _, _ = ct.lqr(A,B,Q,R)
#Ko, _, _ = ct.lqe(A,G,C,Qe,Re)
estim = ct.create_estimator_iosystem(sys, Qe, Re, G=G)
print(type(estim))
ctrl, clsys = ct.create_statefbk_iosystem(sys, Kc, estimator=estim)
print(type(ctrl))
print(type(clsys))