# lqg.py
# Johannes Kaisinger, July 2023
#
# Demonstrate a lqg regulator
#
# [1] 

import control as ct
import numpy as np

# Mass spring damper system
m = 1.; # mass
k = 0.5; # spring
c = 0.1; # damper

A = np.array([[0, 1],[-k/m, -c/m]])
B = np.array([[0], [1/m]])
C = np.eye(2)
D = np.zeros((2,1))

sys_ss = ct.ss(A,B,C,D)
sys = ct.tf(sys_ss)
print(type(sys))

# Controller synthesis
Q = np.eye(2)
R = np.eye(1)
Kc, _, _ = ct.lqr(A,B,Q,R)
ctrl, clsys = ct.create_statefbk_iosystem(sys, Kc, controller_type='linear')
print(type(ctrl))
print(type(clsys))