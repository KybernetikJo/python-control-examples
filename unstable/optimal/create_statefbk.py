import control as ct
import numpy as np

A = [[0, 1], [-0.5, -0.1]]
B = [[0], [1]]
C = np.eye(2)
D = np.zeros((2, 1))
sys = ct.ss(A, B, C, D)

Q = np.eye(2)
R = np.eye(1)

K, _, _ = ct.lqr(sys,Q,R)
ctrl, clsys = ct.create_statefbk_iosystem(sys, K)

