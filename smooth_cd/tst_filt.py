import sys

sys.path.append('../ReadStuff/')
import read_data as rd

import matplotlib.pyplot as plt
import numpy as np
import psd
import cdRLS_smoothing as cdRLS

import matplotlib as mpl
mpl.use('TkAgg')

# Read data
tst = rd.load_test_data_set()
td = 0.2
fs = 1/td

# Parameters
prms = cdRLS.cdRLS_parms("test")

for sig in ['x1', 'x2', 'u1', 'u2', 'T', 'F']:
    for i in range(2):
        for j in range(3):
            (sig_f, g1, g2) = cdRLS.cdRLS_smooth(tst[i][j].ssd[sig], lmda=prms.lmbda, nu=prms.nu[sig], h=prms.h[sig])
            plt.figure()
            plt.plot(tst[i][j].ssd['t'], tst[i][j].ssd[sig], label=sig)
            plt.plot(tst[i][j].ssd['t'], sig_f, label=sig + '_filtered')
            plt.plot(tst[i][j].ssd['t'], g1, label='g1')
            plt.plot(tst[i][j].ssd['t'], g2, label='g2')
            plt.legend()
            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel(sig)
            plt.title(tst[i][j].name)
            plt.tight_layout()
            # plt.savefig("figs/" + dat.name + "/" + dat.name + "_y1.png", dpi=fig_dpi)

            plt.figure()
            f, pd = psd.welch_psd(tst[i][j].ssd[sig], fs)
            f_f, pd_f = psd.welch_psd(sig_f, fs)
            plt.plot(f, pd / (np.max(pd)), label=sig, linewidth=1)
            plt.plot(f_f, pd_f / (np.max(pd_f)), label= sig + "_filtered", linewidth=1)
            plt.plot(0.1 * np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
            plt.xlim([-0.01, 0.2])
            plt.title('Power Spectral Density of ' + sig + ' in ' + tst[i][j].name)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Scaled PSD')
            plt.grid()
            plt.legend()
            # plt.savefig("figs/test_psd/" + sig + ".png", dpi=1200)


for sig in ['y1']:
    for i in range(2):
        for j in range(3):
            (sig_f, g1, g2) = cdRLS.cdRLS_smooth(tst[i][j].iod[sig], lmda=prms.lmbda, nu=prms.nu[sig], h=prms.h[sig])
            plt.figure()
            plt.plot(tst[i][j].iod['t'], tst[i][j].iod[sig], label=sig)
            plt.plot(tst[i][j].iod['t'], sig_f, label=sig + '_filtered')
            plt.plot(tst[i][j].iod['t'], g1, label='g1')
            plt.plot(tst[i][j].iod['t'], g2, label='g2')
            plt.legend()
            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel(sig)
            plt.title(tst[i][j].name)
            plt.tight_layout()
            # plt.savefig("figs/" + dat.name + "/" + dat.name + "_y1.png", dpi=fig_dpi)

            plt.figure()
            f, pd = psd.welch_psd(tst[i][j].iod[sig], fs)
            f_f, pd_f = psd.welch_psd(sig_f, fs)
            plt.plot(f, pd / (np.max(pd)), label=sig, linewidth=1)
            plt.plot(f_f, pd_f / (np.max(pd_f)), label= sig + "_filtered", linewidth=1)
            plt.plot(0.1 * np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
            plt.xlim([-0.01, 0.2])
            plt.title('Power Spectral Density of ' + sig + ' in ' + tst[i][j].name)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Scaled PSD')
            plt.grid()
            plt.legend()
            # plt.savefig("figs/test_psd/" + sig + ".png", dpi=1200)

plt.show()
