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

def cdRLS_withTD(t_skips, y, lmda=0, nu=0, h=0):
    """CD-RLS on data with time-discontinuities
        yt: time series data
       lmda: forgetting factor
       nu: drift parameter
       h: threshold parameter
       return: smoothed data
    """
    n = len(y)
    # initialize variables
    th_hat = np.zeros(n)
    g1     = np.zeros(n)
    g2     = np.zeros(n)
    for i in range(0,len(t_skips)-1):
        th, g_1, g_2 = cdRLS_smooth(y[t_skips[i]:t_skips[i+1]], lmda, nu, h)
        th_hat[t_skips[i]:t_skips[i+1]] = th
        g1[t_skips[i]:t_skips[i+1]] = g_1
        g2[t_skips[i]:t_skips[i+1]] = g_2
    return th_hat, g1, g2

# Example
if __name__=="__main__":
    import sys
    sys.path.append('../ReadStuff/')
    import read_data as rd
    import matplotlib.pyplot as plt
    import matplotlib


    # read data
    dat = rd.Data("truck", 0, 0)
    print(dat.name)
    fig_dpi = 600
    lda = 0.75

    # smoothing y1 data
    t = dat.iod['t']
    t_skips = dat.iod['t_skips']
    y = dat.iod['y1']
    (ys, g1, g2) = cdRLS_withTD(t_skips, y, lmda=lda, nu=10, h=40)
    plt.figure()
    plt.plot(t, y, label='y1')
    plt.plot(t, ys, label='y1_filtered')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('y1 (mol/m^3)')
    plt.title(dat.name)
    plt.tight_layout()
    plt.savefig("figs/"+dat.name+"/"+dat.name+"_y1.png", dpi=fig_dpi)
    

    # smoothing u1 data
    u1 = dat.iod['u1']
    (u1s, g1, g2) = cdRLS_withTD(t_skips, u1, lmda=lda, nu=20, h=40)
    plt.figure()
    plt.plot(t, u1, label='u1')
    plt.plot(t, u1s, label='u1_filtered')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('u1 (mol/m^3)')
    plt.title(dat.name)
    plt.tight_layout()
    plt.savefig("figs/"+dat.name+"/"+dat.name+"_u1.png", dpi=fig_dpi)


    # smoothing u1 data
    u2 = dat.iod['u2']
    (u2s, g1, g2) = cdRLS_withTD(t_skips, u2, lmda=lda, nu=0.8, h=2)
    plt.figure()
    plt.plot(t, u2, label='u2')
    plt.plot(t, u2s, label='u2_filtered')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('u2 (ml/s)')
    plt.title(dat.name)
    plt.tight_layout()
    plt.savefig("figs/"+dat.name+"/"+dat.name+"_u2.png", dpi=fig_dpi)

    # smoothing u1 data
    T = dat.iod['T']
    (Ts, g1, g2) = cdRLS_smooth(T, lmda=lda, nu=20, h=40)
    plt.figure()
    plt.plot(t, T, label='T')
    plt.plot(t, Ts, label='T_filtered')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('T (T -250 deg C)')
    plt.title(dat.name)
    plt.tight_layout()
    plt.savefig("figs/"+dat.name+"/"+dat.name+"_T.png", dpi=fig_dpi)


    # smoothing u1 data
    F = dat.iod['F']
    (Fs, g1, g2) = cdRLS_smooth(F, lmda=lda, nu=150, h=300)
    plt.figure()
    plt.plot(t, F, label='F')
    plt.plot(t, Fs, label='F_filtered')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('F (g/s)')
    plt.title(dat.name)
    plt.tight_layout()
    plt.savefig("figs/"+dat.name+"/"+dat.name+"_F.png", dpi=fig_dpi)

#    # smoothing x2 data
#    t = dat.ssd['t']
#    x2 = dat.ssd['x2']
#    (x2s, g1, g2) = cdRLS_smooth(x2, lmda=0.99, nu=0.03, h=0.1)
#    plt.figure()
#    plt.plot(t, x2, label='x2')
#    plt.plot(t, x2s, label='x2_smoothed')
#    plt.plot(t, g1, label='g1')
#    plt.plot(t, g2, label='g2')
#    plt.legend()
#    plt.grid()
#    plt.xlabel('Time (s)')
#    plt.ylabel('x_2 (mol/m^3)')
#    plt.title(dat.name)
#    plt.tight_layout()
#    plt.savefig("figs/"+dat.name+"/"+dat.name+"_x2.png", dpi=fig_dpi)

    # plt.close('all')
    plt.show()
