import sys
sys.path.append('../ReadStuff/')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
from scipy.optimize import lsq_linear
import read_data as rd
import phi_alg as phi


class tst_parm_id:
    """Parameter identification classs"""
    def __init__(self, dat):
        """Get the id started using the data object"""
        self.dat = dat
        self.get_data()
        self.Phi_NOx_mat = self.gen_Phi_NOx_mat()
        self.y_mat = self.gen_y_mat()
        self.theta = self.solve_lsq_prob()

    def solve_lsq_prob(self):
        """Solve the linear least squares"""
        A = self.Phi_NOx_mat
        b = self.y_mat.flatten()
        lb = np.zeros(10)
        ub = np.Inf*np.ones(10)
        sol = lsq_linear(A, b, bounds=(lb,ub))
        return sol.x

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
            x1kp1 = self.x1[i+1]
            x1k = self.x1[i]
            u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm = self.get_curr_prev_inputs(i)
            dNOx_kp1 = x1kp1 - u1k
            f_phi_1_k = phi.calc_f_phi_1_k(u2k, Tk, Fk, u2m, Tm, Fm)
            dNOx_k = (x1k - u1m) * f_phi_1_k
            y[i-1, 0] = dNOx_kp1 - dNOx_k
        return y

    def get_data(self):
        self.t = dat.ssd['t']
        self.x1 = dat.ssd['x1']
        self.u1 = dat.ssd['u1']
        self.u2 = dat.ssd['u2']
        self.T = dat.ssd['T']
        self.F = dat.ssd['F']
        self.N = len(self.t)

    def gen_Phi_NOx_mat(self):
        """Generate the A matrix for lsq_linear"""
        Phi_NOx_mat = np.zeros((self.N-2, 10))
        for i in range(1, self.N-1):
            x1k = self.x1[i]
            u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm = self.get_curr_prev_inputs(i)
            phi_NOx = phi.cnstrct_Phi_NOx(x1k, u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm)
            Phi_NOx_mat[i-1, :] = phi_NOx[:, 0].flatten()
        return Phi_NOx_mat


if __name__ == '__main__':

    dat = rd.Data("test", 0, 2)
    prm_id = tst_parm_id(dat)
    print(np.round(prm_id.theta, 6))
