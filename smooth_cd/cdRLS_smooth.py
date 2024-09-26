import numpy as np

def cdRLS_smooth(y, lmda=0, nu=0, h=0):
    """yt: time series data
       lmda: forgetting factor
       nu: drift parameter
       h: threshold parameter
       return: smoothed data
    """
    n = len(y)
    # initialize variables
    th_hat = np.zeros(n)
    eps    = np.zeros(n)
    s1     = np.zeros(n)
    s2     = np.zeros(n)
    g1     = np.zeros(n)
    g2     = np.zeros(n)
    th_hat[0] = y[0]
    for t in range(1,n):
        # update variables
        eps[t] = y[t] - th_hat[t-1]
        s1[t] = eps[t]
        s2[t] = -eps[t]
        g1[t] = np.max([g1[t-1] + s1[t] - nu, 0])
        g2[t] = np.max([g2[t-1] + s2[t] - nu, 0])
        if (g1[t] > h) or (g2[t] > h):  # change detection
            th_hat[t] = y[t]            # reset rls estimate
            g1[t] = 0
            g2[t] = 0
        else:
            th_hat[t] = lmda * th_hat[t - 1] + (1 - lmda) * y[t]        # RLS estimate
    return th_hat, g1, g2


# Example
if __name__=="__main__":
    import sys
    sys.path.append('../ReadStuff/')
    import read_data as rd
    import matplotlib.pyplot as plt

    # read data
    dg_rmc = rd.Data("test", 0, 2)

    # smoothing y1 data
    t = dg_rmc.iod['t']
    y = dg_rmc.iod['y1']
    (ys, g1, g2) = cdRLS_smooth(y, lmda=0.95, nu=0.05, h=0.1)
    plt.figure()
    plt.plot(t, y, label='y1')
    plt.plot(t, ys, label='y1_smoothed')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()

    # smoothing u1 data
    u1 = dg_rmc.iod['u1']
    (u1s, g1, g2) = cdRLS_smooth(u1, lmda=0.95, nu=1, h=3)
    plt.figure()
    plt.plot(t, u1, label='u1')
    plt.plot(t, u1s, label='u1_smoothed')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()
    
    # smoothing u1 data
    u2 = dg_rmc.iod['u2']
    (u2s, g1, g2) = cdRLS_smooth(u2, lmda=0.95, nu=0.1, h=0.2)
    plt.figure()
    plt.plot(t, u2, label='u2')
    plt.plot(t, u2s, label='u2_smoothed')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()

    # smoothing u1 data
    T = dg_rmc.iod['T']
    (Ts, g1, g2) = cdRLS_smooth(T, lmda=0.99, nu=0.001, h=0.01)
    plt.figure()
    plt.plot(t, T, label='T')
    plt.plot(t, Ts, label='T_smoothed')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()

    # smoothing u1 data
    F = dg_rmc.iod['F']
    (Fs, g1, g2) = cdRLS_smooth(F, lmda=0.95, nu=10, h=50)
    plt.figure()
    plt.plot(t, F, label='F')
    plt.plot(t, Fs, label='F_smoothed')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()

    # smoothing x2 data
    x2 = dg_rmc.ssd['x2']
    (x2s, g1, g2) = cdRLS_smooth(x2, lmda=0.99, nu=0.03, h=0.1)
    plt.figure()
    plt.plot(t, x2, label='x2')
    plt.plot(t, x2s, label='x2_smoothed')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()

    plt.show()

