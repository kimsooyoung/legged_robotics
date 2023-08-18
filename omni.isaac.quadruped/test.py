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

modified_contacts = [True, True, True, True]
F_min = 0
F_max = 250.0
mu = 0.2

linearMatrix = np.zeros([20, 12])
lowerBound = np.zeros(20)
upperBound = np.zeros(20)
for i in range(4):
    # extract F_zi
    linearMatrix[i, 2 + i * 3] = 1.0
    # friction pyramid
    # 1. F_xi < uF_zi
    linearMatrix[4 + i * 4, i * 3] = 1.0
    linearMatrix[4 + i * 4, 2 + i * 3] = -mu
    lowerBound[4 + i * 4] = -np.inf
    # 2. -F_xi > uF_zi
    linearMatrix[4 + i * 4 + 1, i * 3] = -1.0
    linearMatrix[4 + i * 4 + 1, 2 + i * 3] = -mu
    lowerBound[4 + i * 4 + 1] = -np.inf
    # 3. F_yi < uF_zi
    linearMatrix[4 + i * 4 + 2, 1 + i * 3] = 1.0
    linearMatrix[4 + i * 4 + 2, 2 + i * 3] = -mu
    lowerBound[4 + i * 4 + 2] = -np.inf
    # 4. -F_yi > uF_zi
    linearMatrix[4 + i * 4 + 3, 1 + i * 3] = -1.0
    linearMatrix[4 + i * 4 + 3, 2 + i * 3] = -mu
    lowerBound[4 + i * 4 + 3] = -np.inf

    c_flag = 1.0 if modified_contacts[i] else 0.0
    lowerBound[i] = c_flag * F_min
    upperBound[i] = c_flag * F_max

print(f'linearMatrix: {linearMatrix}')
print(f'lowerBound: {lowerBound}')
print(f'upperBound: {upperBound}')

"""
linearMatrix: 
FL_x, FL_y, FL_z,FR_x,FR_y,FR_z,RL_x,RL_y,RL_z,RR_x,RR_y,RR_z
[[ 0.   0.   1.   0.   0.   0.   0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.   0.   0.   1.   0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   1.   0.   0.   0. ]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   1. ]
 [ 1.   0.  -0.2  0.   0.   0.   0.   0.   0.   0.   0.   0. ]
 [-1.   0.  -0.2  0.   0.   0.   0.   0.   0.   0.   0.   0. ]
 [ 0.   1.  -0.2  0.   0.   0.   0.   0.   0.   0.   0.   0. ]
 [ 0.  -1.  -0.2  0.   0.   0.   0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.   1.   0.  -0.2  0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.  -1.   0.  -0.2  0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.   0.   1.  -0.2  0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.   0.  -1.  -0.2  0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.   0.   0.   0.   1.   0.  -0.2  0.   0.   0. ]
 [ 0.   0.   0.   0.   0.   0.  -1.   0.  -0.2  0.   0.   0. ]
 [ 0.   0.   0.   0.   0.   0.   0.   1.  -0.2  0.   0.   0. ]
 [ 0.   0.   0.   0.   0.   0.   0.  -1.  -0.2  0.   0.   0. ]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0.   1.   0.  -0.2]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0.  -1.   0.  -0.2]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   1.  -0.2]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  -1.  -0.2]]
lowerBound: [  0.   0.   0.   0. -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf
 -inf -inf -inf -inf -inf -inf]
upperBound: [250. 250. 250. 250.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.]
0 <= FL_z <= 250
0 <= FR_z <= 250
0 <= RL_z <= 250
0 <= RR_z <= 250
-inf <= FL_x - 0.2*FL_z <= 0
-inf <= -FL_x - 0.2*FL_z <= 0
therefore -0.2*FL_z <= FL_x <= 0.2*FL_z

-0.2*FL_z <= FL_y <= 0.2*FL_z
-0.2*FR_z <= FR_x <= 0.2*FR_z
-0.2*FR_z <= FR_y <= 0.2*FR_z
-0.2*RL_z <= RL_x <= 0.2*RL_z
-0.2*RL_z <= RL_y <= 0.2*RL_z
-0.2*RR_z <= RR_x <= 0.2*RR_z
-0.2*RR_z <= RR_y <= 0.2*RR_z
"""
