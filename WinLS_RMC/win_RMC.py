import sys
sys.path.append('../ReadStuff/')
sys.path.append('../smooth_cd/')

import numpy as np
from scipy.optimize import lsq_linear, nnls
import pickle

import read_data as rd
import cdRLS_smoothing as cdRLS

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')


def phi1(u1, u3, u4):
    """ Calculates the phi matrix"""
    n = len(u1)
    phi_1 = np.zeros([n,5])
    for i in range(n):
        phi_1[i, 0] = u1[i] * u3[i]**2
        phi_1[i, 1] = u1[i] * u3[i] * u4[i]
        phi_1[i, 2] = u1[i] * (-u3[i])
        phi_1[i, 3] = u1[i] * (u4[i])
        phi_1[i, 4] = -u1[i]
    return phi_1

f_name = 'bt_age.pkl'
dg_rmc = rd.Data("test", 0, 2)
# data
t  = dg_rmc.ssd['t']
x1 = dg_rmc.ssd['x1']
u1 = dg_rmc.ssd['u1']
u3 = dg_rmc.ssd['T']
u4 = dg_rmc.ssd['F']
#
wl = 1000
N = len(t)
n = int(np.ceil(N/wl))
bt = np.zeros([n, 5])
r = np.zeros(n)

for i in range(n):
    if i == n-1:
        b = - u1[wl*i:-1] + x1[wl*i+1:]
        A = phi1(u1[wl*i:-1], u3[wl*i:-1], u4[wl*i:-1])
    else:
        b = - u1[wl*i: wl*(i+1)] + x1[(wl*i)+1 : wl*(i+1)+1]
        A = phi1(u1[wl*i: wl*(i+1)], u3[wl*i: wl*(i+1)], u4[wl*i: wl*(i+1)])
    res = lsq_linear(A, b)
    bt[i, :] = res.x

pickle.dump(bt, open(f_name, 'wb'))
#=======================================================================================================================
# Prediction error

x1p = np.zeros(np.size(x1))
dx = np.zeros(np.size(x1))
x1p[0] = x1[0]

for i in range(0, N-1):
    j = int(np.floor(i/wl))
    if j>=(n-1):
        j = n-1
    btj = np.matrix(bt[j, :]).T
    A  = phi1([u1[i]], [u3[i]], [u4[i]])
    dx[i] = (A @ btj)[0,0]
    x1p[i+1] = u1[i] + dx[i]

# #smooth the results
# prm = cdRLS.cdRLS_parms("test")
# x1ps, g1, g2 = cdRLS.cdRLS_smooth(x1p, prm.lmbda, prm.nu['x1'], prm.h['x1'])

plt.figure()
plt.plot(t, x1, linewidth=1)
plt.plot(t, x1p, linewidth=1)
# plt.plot(t, dx, linewidth=1)
# plt.plot(t, u1, linewidth=1)
plt.xlabel("Time")
plt.ylabel('[NOx] out mol/m^3')
plt.legend(["[NOx] out", "[NOx] out predicted", "dx", "u1"])
plt.title(dg_rmc.name)
plt.grid()
# plt.savefig("./figs/"+dg_rmc.name+".png")
# plt.show()

plt.figure()
plt.plot(t, x1-x1p, linewidth=1)
# plt.plot(t, dx, linewidth=1)
# plt.plot(t, u1, linewidth=1)
plt.xlabel("Time")
plt.ylabel('Prediction error in [NOx] out mol/m^3')
plt.title(dg_rmc.name)
plt.grid()
# plt.savefig("./figs/"+dg_rmc.name+".png")
# plt.show()


for i in range(5):
    plt.figure()
    plt.plot(bt[:, i], 'x')
    plt.grid()
    plt.legend(["bt_"+str(i)])
###
###
###
# Prediction error
dg_rmc = rd.Data("test", 0, 1)
# data
t  = dg_rmc.ssd['t']
x1 = dg_rmc.ssd['x1']
u1 = dg_rmc.ssd['u1']
u3 = dg_rmc.ssd['T']
u4 = dg_rmc.ssd['F']
#
N = len(t)

x1p = np.zeros(np.size(x1))
dx = np.zeros(np.size(x1))
x1p[0] = x1[0]

for i in range(0, N-1):
    j = int(np.floor(i/wl))
    if j>=(n-1):
        j = n-1
    btj = np.matrix(bt[j, :]).T
    A  = phi1([u1[i]], [u3[i]], [u4[i]])
    dx[i] = (A @ btj)[0,0]
    x1p[i+1] = u1[i] + dx[i]

# #smooth the results
# prm = cdRLS.cdRLS_parms("test")
# x1ps, g1, g2 = cdRLS.cdRLS_smooth(x1p, prm.lmbda, prm.nu['x1'], prm.h['x1'])

plt.figure()
plt.plot(t, x1, linewidth=1)
plt.plot(t, x1p, linewidth=1)
# plt.plot(t, dx, linewidth=1)
# plt.plot(t, u1, linewidth=1)
plt.xlabel("Time")
plt.ylabel('[NOx] out mol/m^3')
plt.legend(["[NOx] out", "[NOx] out predicted", "dx", "u1"])
plt.title(dg_rmc.name)
plt.grid()
# plt.savefig("./figs/"+dg_rmc.name+".png")
# plt.show()

plt.figure()
plt.plot(t, x1-x1p, linewidth=1)
# plt.plot(t, dx, linewidth=1)
# plt.plot(t, u1, linewidth=1)
plt.xlabel("Time")
plt.ylabel('Prediction error in [NOx] out mol/m^3')
plt.title(dg_rmc.name)
plt.grid()
# plt.savefig("./figs/"+dg_rmc.name+".png")
# plt.show()




plt.show()