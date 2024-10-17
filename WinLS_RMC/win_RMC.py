import sys
sys.path.append('../ReadStuff/')
import read_data as rd
import numpy as np
from scipy.optimize import lsq_linear, nnls

def phi1(u1, u2, u3, u4):
    """ Calculates the phi matrix"""
    n = len(u1)
    phi_1 = np.zeros([n,3])
    for i in range(n):
        phi_1[i, 0] = u1[i] * (u2[i]/u4[i])
        phi_1[i, 1] = u1[i] * (1/u4[i])
        phi_1[i, 2] = u1[i] * (1/u3[i]*u4[i])
    return phi_1

dg_rmc = rd.Data("test", 0, 2)
# data
t = dg_rmc.ssd['t']
x1 = dg_rmc.ssd['x1']
u1 = dg_rmc.ssd['u1']
u2 = dg_rmc.ssd['u2']
u3 = dg_rmc.ssd['T']
u4 = dg_rmc.ssd['F']
#
wl = 10
N = len(t)
n = int(np.ceil(N/wl))
bt = np.zeros([n, 3])
r = np.zeros(n)
btp = np.zeros([n, 3])

for i in range(n):
    if i == n-1:
        b = u1[i:] - x1[i:]
        A = phi1(u1[i:], u2[i:], u3[i:], u4[i:])
    else:
        b = u1[i:i + wl] - x1[i:i + wl]
        A = phi1(u1[i:i + wl], u2[i:i + wl], u3[i:i + wl], u4[i:i + wl])
    res = lsq_linear(A, b)
    bt[i, :] = res.x
    btp[i,:], r[i] = nnls(A, b)






