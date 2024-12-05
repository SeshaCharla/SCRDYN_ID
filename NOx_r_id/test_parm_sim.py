import sys
sys.path.append('../ReadStuff/')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import read_data as rd
import phi_alg as phi


class tst_parm_sim:
    """Parameter identification classs"""
    def __init__(self, dat, theta, n_steps):
        """Get the id started using the data object"""
        self.dat = dat
        self.theta = np.reshape(theta, [10, 1])
        self.n_steps = n_steps
        self.get_data()
        self.x1_sim = self.sim_x1()
        self.error = self.x1 - self.x1_sim

    def sim_x1(self):
        """Simulate x1 using the input data"""
        x1_sim = np.zeros(self.N)
        for i in range(1, self.N):
            if i % self.n_steps == 0:
                x1_sim[i] = self.x1[i]
            else:
                u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm = self.get_curr_prev_inputs(i)
                x1k = x1_sim[i-1]
                phi_NOx = phi.cnstrct_Phi_NOx(x1k, u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm)
                f_phi_1_k = phi.calc_f_phi_1_k(u2k, Tk, Fk, u2m, Tm, Fm)
                dNOx_k = (x1k - u1m) * f_phi_1_k
                x1_sim[i] = u1k + dNOx_k + (phi_NOx.T @ self.theta)[0, 0]
        return x1_sim

    def get_curr_prev_inputs(self, i):
        u1k = self.u1[i]
        u2k = self.u2[i]
        Tk = self.T[i]
        Fk = self.F[i]
        u1m = self.u1[i - 1]
        u2m = self.u2[i - 1]
        Tm = self.T[i - 1]
        Fm = self.F[i - 1]
        return u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm

    def get_data(self):
        self.t = dat.ssd['t']
        self.x1 = dat.ssd['x1']
        self.u1 = dat.ssd['u1']
        self.u2 = dat.ssd['u2']
        self.T = dat.ssd['T']
        self.F = dat.ssd['F']
        self.N = len(self.t)


if __name__ == '__main__':

    dat = rd.Data("test", 0, 2)
    theta = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.48100000e-03, 0.00000000e+00,
                      9.47000000e-04, 2.38150000e-01, 0.00000000e+00, 0.00000000e+00, 2.01389966e+03])
    prm_sim = tst_parm_sim(dat, theta, 1000)

    plt.figure()
    plt.plot(prm_sim.t, prm_sim.x1_sim)
    plt.plot(prm_sim.t, prm_sim.x1)
    plt.show()
