import sys
sys.path.append('../ReadStuff/')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import read_data as rd
import phi_alg as phi
import pickle


class tst_parm_sim:
    """Parameter identification classs"""
    def __init__(self, dat, theta, n_steps):
        """Get the id started using the data object"""
        self.n_parms = 8
        self.dat = dat
        self.theta = np.reshape(theta, [self.n_parms, 1])
        self.n_steps = n_steps
        self.offset = 0
        self.get_data()
        self.eta_sim = self.sim_eta()
        self.error = self.eta - self.eta_sim

    def sim_eta(self):
        """Simulate x1 using the input data"""
        eta_sim = np.zeros(self.N)
        eta_sim[0] = self.eta[0]
        for i in range(1, self.N):
            if i % self.n_steps == 0:
                eta_sim[i] = self.eta[i]
            else:
                u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm = self.get_curr_prev_inputs(i)
                etak = eta_sim[i-1]
                phi_NOx = phi.cnstrct_Phi_NOx(etak, u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm)
                f_phi_1_k = phi.calc_f_phi_1_k(u1k, Tk, Fk, u1m, Tm, Fm)
                dNOx_k = etak * f_phi_1_k
                eta_sim[i] = dNOx_k + (phi_NOx.T @ self.theta)[0, 0]
        return eta_sim

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
        self.t   = (self.dat.ssd['t'])[self.offset:-1-self.offset]
        self.u1  = (self.dat.ssd['u1'])[self.offset:-1-self.offset]
        self.u2  = (self.dat.ssd['u2'])[self.offset:-1-self.offset]
        self.T   = (self.dat.ssd['T'])[self.offset:-1-self.offset]
        self.F   = (self.dat.ssd['F'])[self.offset:-1-self.offset]
        self.eta = (self.dat.ssd['eta'])[self.offset:-1-self.offset]
        self.N  = len(self.t)

if __name__ == '__main__':

    dat = rd.Data("test", 0, 2)
    theta = pickle.load(open("prm_id.pkl", 'rb'))
    print(theta)
    prm_sim = tst_parm_sim(dat, theta, np.Inf)

    plt.figure()
    plt.plot(prm_sim.t, prm_sim.eta_sim)
    plt.plot(prm_sim.t, prm_sim.eta)
    plt.show()
