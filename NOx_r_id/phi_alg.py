import numpy as np

def cnstrct_Phi_NOx(x1k, u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm):
    """k is current time and
    m = k-1"""
    # Correcting for zeros in F and u2 this is valid for test-cell data
    F_datum = lambda F : F if F > 25 else 25
    u2_datum = lambda u2 : u2 if u2 > 0.025 else 0.025
    Fk = F_datum(Fk)
    Fm = F_datum(Fm)
    u2k = u2_datum(u2k)
    u2m = u2_datum(u2m)
    #
    phi_k = np.matrix([[Tk], [1]])
    phi_m = np.matrix([[Tm], [1]])
    phi_tau_k = (1/Fk) * phi_k
    phi_tau_m = (1/Fm) * phi_m
    phi_ur_m = u2m * phi_tau_m
    phi_1_k = -u1k * phi_tau_k
    f_phi_1_k = ((u2k)/(u1m)) * ((Fm)/(Fk)) * ((Tk*Tm + 1)/(Tm**2 + 1))
    dNOx = (x1k - u1m)*f_phi_1_k
    phi_f1_k = dNOx * np.concatenate([phi_ur_m, phi_m, u1m*phi_m], axis=0)
    phi_gamma1 = np.kron(phi_1_k, phi_ur_m)
    phi_nox = np.concatenate([-phi_f1_k, phi_gamma1], axis=0)
    return phi_nox


if __name__ == '__main__':
    xk = 15
    u1k = 20
    u2k = 1
    Tk = 50
    Fk = 300
    u1m = u1k-3
    u2m = u1m - 0.4
    Tm = 30
    Fm = 275
    print(cnstrct_Phi_NOx(xk, u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm))