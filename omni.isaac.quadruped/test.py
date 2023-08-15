import osqp
import numpy as np
import scipy as sp
from scipy import sparse

N = 10
Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
QN = Q
R = 0.1*sparse.eye(4)

# (120, 120)
# (12, 12)
# (40, 40)
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# (172, 172)
print([sparse.kron(sparse.eye(N), Q), QN, sparse.kron(sparse.eye(N), R)])