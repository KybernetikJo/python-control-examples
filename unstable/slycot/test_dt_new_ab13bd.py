import numpy as np
import slycot
print(slycot.__version__)

A = np.array([[0.1, 0.9],[0.9, 0.1]])
B = np.array([[0.],[1.]])
C = np.eye(2)
D = np.zeros((2,1))

dico = 'D'
jobn = 'H'

n, m = B.shape
p, _ = C.shape

print(slycot.ab13bd(dico, jobn, n, m, p, A, B, C, D))
print(slycot.ab13bd(dico, jobn, n, m, p, A, B, C, D))
print(slycot.ab13bd(dico, jobn, n, m, p, A, B, C, D))