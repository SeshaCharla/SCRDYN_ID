import numpy as np

def F_datum(F):
    F_min = 25
    if F > F_min:
        return  F
    else:
        return F_min

def u1_datum(u1):
    u1_min = 0.001
    if u1 > u1_min:
        return u1
    else:
        return u1_min


def cnstrct_Phi_NOx(etak, u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm):
    """k is current time and
    m = k-1"""
    # Correcting for zeros in F and u2 this is valid for test-cell data
    Fk = F_datum(Fk)
    Fm = F_datum(Fm)
    u1k = u1_datum(u1k)
    u1m = u1_datum(u1m)
    #
    phi_k = np.matrix([[Tk], [1]])
    phi_m = np.matrix([[Tm], [1]])
    phi_tau_k = (1/Fk) * phi_k
    phi_tau_m = (1/Fm) * phi_m
    phi_ur_m = u2m * phi_tau_m
    phi_1_k = u1k * phi_tau_k
    f_phi_1_k = calc_f_phi_1_k(u1k, Tk, Fk, u1m, Tm, Fm)
    dNOx = etak*f_phi_1_k
    phi_f1_k = dNOx * np.concatenate([phi_ur_m, phi_m, u1m*phi_m], axis=0)
    phi_gamma1 = (u1k/Fk)*(u2m/Fm)*np.matrix([[Tk], [1]])
    phi_nox = np.concatenate([-phi_f1_k, phi_gamma1], axis=0)
    return phi_nox


def calc_f_phi_1_k(u1k, Tk, Fk, u1m, Tm, Fm):
    """Calculate f_phi_1(k)"""
    # Correcting for zeros in F and u2 this is valid for test-cell data
    Fk = F_datum(Fk)
    Fm = F_datum(Fm)
    u1k = u1_datum(u1k)
    u1m = u1_datum(u1m)
    f_phi_1_k = ((u1k)/(u1m)) * ((Fm)/(Fk)) * ((Tk*Tm + 1)/(Tm**2 + 1))
    return f_phi_1_k


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
    phi_NOx = cnstrct_Phi_NOx(xk, u1k, u2k, Tk, Fk, u1m, u2m, Tm, Fm)
    print(phi_NOx)