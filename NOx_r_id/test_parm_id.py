import sys
sys.path.append('../ReadStuff/')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
from scipy.optimize import lsq_linear
import read_data as rd
import phi_alg as phi
import pickle


class tst_parm_id:
    """Parameter identification classs"""
    def __init__(self, dat):
        """Get the id started using the data object"""
        self.n_parms = 8
        self.dat = dat
        self.offset = 0
        self.get_data()
        self.Phi_NOx_mat = self.gen_Phi_NOx_mat()
        self.y_mat = self.gen_y_mat()
        self.theta = self.solve_lsq_prob()

    def solve_lsq_prob(self):
        """Solve the linear least squares"""
        A = self.Phi_NOx_mat
        b = self.y_mat.flatten()
        lb = np.zeros(self.n_parms)
        ub = 1000*np.ones(self.n_parms)
        sol = lsq_linear(A, b)   #, bounds=(lb,ub))
        print(sol)
        return np.round(sol.x, 6)

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

    def gen_y_mat(self):
        """Generate the b matrix for lsq_linear"""
        y = np.zeros([self.N-2, 1])
        for i in range(1, self.N-1):
            etakp1 = self.eta[i+1]
            etak = self.eta[i]
            u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm = self.get_curr_prev_inputs(i)
            f_phi_1_k = phi.calc_f_phi_1_k(u1k, Tk, Fk, u1m, Tm, Fm)
            y[i-1, 0] = etakp1 - (etak * f_phi_1_k)
        return y

    def get_data(self):
        self.t   = (self.dat.ssd['t'])[self.offset:-1-self.offset]
        self.u1  = (self.dat.ssd['u1'])[self.offset:-1-self.offset]
        self.u2  = (self.dat.ssd['u2'])[self.offset:-1-self.offset]
        self.T   = (self.dat.ssd['T'])[self.offset:-1-self.offset]
        self.F   = (self.dat.ssd['F'])[self.offset:-1-self.offset]
        self.eta = (self.dat.ssd['eta'])[self.offset:-1-self.offset]
        self.N  = len(self.t)

    def gen_Phi_NOx_mat(self):
        """Generate the A matrix for lsq_linear"""
        Phi_NOx_mat = np.zeros((self.N-2, self.n_parms))
        for i in range(1, self.N-1):
            etak = self.eta[i]
            u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm = self.get_curr_prev_inputs(i)
            phi_NOx = phi.cnstrct_Phi_NOx(etak, u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm)
            Phi_NOx_mat[i-1, :] = phi_NOx[:, 0].flatten()
        return Phi_NOx_mat


if __name__ == '__main__':

    dat = rd.Data("test", 0, 2)
    prm_id = tst_parm_id(dat)
    pickle.dump(np.round(prm_id.theta, 4), open('prm_id.pkl', 'wb'))
    print(np.round(prm_id.theta, 2))
