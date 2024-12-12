import sys
sys.path.append('../ReadStuff/')

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
from scipy.optimize import lsq_linear
import read_data as rd

def phi_NOx_yi(xkp1, new_toup, old_toup):
    # (x1_k, u1_k, Uinj_k, Tk, Fk), (x1_m, u1_m, Uinj_m, Tm, Fm)):
    # Assigning the variables
    null_F = 200
    x1_k = new_toup[0]
    u1_k = new_toup[1]
    Uinj_k = new_toup[2]
    Tk = new_toup[3]
    Fk = new_toup[4] + null_F
    x1_m = old_toup[0]
    u1_m = old_toup[1]
    Uinj_m = old_toup[2]
    Tm = old_toup[3]
    Fm = old_toup[4] + null_F
    ###
    phi_tau_k = np.matrix([[Tk/(Fk)], [1/(Fk)]])
    phi_tau_m = np.matrix([[Tm/(Fm)], [1/(Fm)]])
    # phi_tauUR_k = (Uinj_k/(Fk)) * phi_tau_k
    phi_tauUR_m = (Uinj_m/(Fm)) * phi_tau_m
    phi_1_k = -u1_k*phi_tau_k
    phi_1_m = -u1_m*phi_tau_m
    f_phi1_k = (phi_1_k.T @ np.linalg.pinv(phi_1_m.T))[0, 0]
    phi_f1 = (x1_k-u1_m) * f_phi1_k * np.concatenate((phi_tauUR_m, phi_tau_m, phi_1_m), axis=0)
    phi_gam1 = np.kron(phi_1_k, phi_tauUR_m)
    phi_nox = np.concatenate((-phi_f1, phi_gam1), axis=0)
    yi = (xkp1 - u1_k) - (x1_k - u1_m)*f_phi1_k
    return phi_nox, yi

def construct_test_Phi_y(dat):
    # Test data
    t = dat.ssd['t']
    x1 = dat.ssd['x1']
    u1 = dat.ssd['u1']
    u2 = dat.ssd['u2']
    T = dat.ssd['T']
    F = dat.ssd['F']
    N = len(t)
    Phi = np.zeros((N-2, 10))
    y = np.zeros((N-2, 1))
    old_touple = (x1[0], u1[0], u2[0], T[0], F[0])
    for j in range(1, N-1):
        new_touple =  (x1[j], u1[j], u2[j], T[j], F[j])
        phi_nox, yi = phi_NOx_yi(x1[j+1],new_touple, old_touple)
        Phi[j-1,:] = (phi_nox.T)[:,:]
        y[j-1, 0] = yi
        old_touple = new_touple
    return Phi, y


def xkp1_NOx(xkp1, new_toup, old_toup, thetas):
    # same regression vector
    phi, ym = phi_NOx_yi(xkp1, new_toup, old_toup)
    dxkp1 = (phi.T @ thetas)[0, 0]
    return -ym + dxkp1


if __name__ == '__main__':
    import cdRLS_smoothing as cdRLS
    import pickle
    dat = rd.Data("test", 0,2)
    Phi, y = construct_test_Phi_y(dat)
    thetas = np.linalg.pinv(Phi)@y
    est_name = dat.name
    pickle.dump(thetas, open("./pkl_parms/"+est_name+".pkl", "wb"))

    # validation
    dat = rd.Data("test", 0,2)
    pred_name = dat.name
    t = dat.ssd['t']
    x1 = dat.ssd['x1']
    u1 = dat.ssd['u1']
    u2 = dat.ssd['u2']
    T = dat.ssd['T']
    F = dat.ssd['F']
    N = len(t)
    xkp1 = np.zeros(N)
    xkp1[0] = x1[0]
    xkp1[1] = x1[1]
    old_touple = [x1[0], u1[0], u2[0], T[0], F[0]]
    for i in range(2, N):
        new_touple = (x1[i-1], u1[i-1], u2[i-1], T[i-1], F[i-1])
        xkp1[i] = xkp1_NOx(0,new_touple, old_touple, thetas)
        old_touple = new_touple

    #smooth the results
    prm = cdRLS.cdRLS_parms("test")
    x1ps, g1, g2 = cdRLS.cdRLS_smooth(xkp1, prm.lmbda, prm.nu['x1'], prm.h['x1'])

    plt.figure()
    plt.plot(t, x1)
    plt.plot(t, xkp1)
    plt.grid(True)
    plt.legend(['Measured NOx','Predicted NOx'])
    plt.xlabel('Time (s)')
    plt.ylabel('NOx mol/m^3')
    plt.title(est_name + " parameters on " + pred_name + " data")
    plt.savefig("./figs/"+est_name + "_" + pred_name + ".png")
    plt.show()
