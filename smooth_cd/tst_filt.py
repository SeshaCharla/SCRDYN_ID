import sys

sys.path.append('../ReadStuff/')
import read_data as rd

import matplotlib.pyplot as plt
import numpy as np
import psd
import cdRLS_smoothing as cdRLS

import matplotlib as mpl
mpl.use('TkAgg')

import pathlib as pth

# Read data
dat = rd.load_test_data_set()
td = 0.2
fs = 1/td

# Parameters
prms = cdRLS.cdRLS_parms("test")

for sig in ['x1', 'x2', 'u1', 'u2', 'T', 'F']:
    for i in range(2):
        for j in range(3):
            (sig_f, g1, g2) = cdRLS.cdRLS_smooth(dat[i][j].ssd[sig], lmda=prms.lmbda, nu=prms.nu[sig], h=prms.h[sig])
            plt.figure()
            plt.plot(dat[i][j].ssd['t'], dat[i][j].ssd[sig], label=sig)
            plt.plot(dat[i][j].ssd['t'], sig_f, label=sig + '_filtered')
            plt.plot(dat[i][j].ssd['t'], g1, label='g1')
            plt.plot(dat[i][j].ssd['t'], g2, label='g2')
            plt.legend()
            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel(sig)
            plt.title(dat[i][j].name)
            plt.tight_layout()
            # Saving the figure
            direct = pth.Path("figs/tst_filt/" + dat[i][j].name)
            direct.mkdir(parents=True, exist_ok=True)
            plt.savefig("figs/tst_filt/" + dat[i][j].name + "/"+ sig +".png", dpi=150)
            plt.close()
            print(dat[i][j].name + " - [max, min] "+sig+" = [{}, {}]".format(np.round(np.max(sig_f), 2),
                                                                             np.round(np.min(sig_f), 2))
                  )

            plt.figure()
            f, pd = psd.welch_psd(dat[i][j].ssd[sig], fs)
            f_f, pd_f = psd.welch_psd(sig_f, fs)
            plt.plot(f, pd / (np.max(pd)), label=sig, linewidth=1)
            plt.plot(f_f, pd_f / (np.max(pd_f)), label= sig + "_filtered", linewidth=1)
            plt.plot(0.1 * np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
            plt.xlim([-0.01, 0.2])
            plt.title('Power Spectral Density of ' + sig + ' in ' + dat[i][j].name)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Scaled PSD')
            plt.grid()
            plt.legend()
            plt.savefig("figs/tst_filt/" + dat[i][j].name + "/" + sig + "_psd.png", dpi=150)
            plt.close()


for sig in ['y1']:
    for i in range(2):
        for j in range(3):
            (sig_f, g1, g2) = cdRLS.cdRLS_smooth(dat[i][j].iod[sig], lmda=prms.lmbda, nu=prms.nu[sig], h=prms.h[sig])
            plt.figure()
            plt.plot(dat[i][j].iod['t'], dat[i][j].iod[sig], label=sig)
            plt.plot(dat[i][j].iod['t'], sig_f, label=sig + '_filtered')
            plt.plot(dat[i][j].iod['t'], g1, label='g1')
            plt.plot(dat[i][j].iod['t'], g2, label='g2')
            plt.legend()
            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel(sig)
            plt.title(dat[i][j].name)
            plt.tight_layout()
            plt.savefig("figs/tst_filt/" + dat[i][j].name + "/" + sig + ".png", dpi=150)
            plt.close()
            print(dat[i][j].name + " - [max, min] "+sig+" = [{}, {}]".format(np.round(np.max(sig_f), 2),
                                                                             np.round(np.min(sig_f), 2))
                  )

            plt.figure()
            f, pd = psd.welch_psd(dat[i][j].iod[sig], fs)
            f_f, pd_f = psd.welch_psd(sig_f, fs)
            plt.plot(f, pd / (np.max(pd)), label=sig, linewidth=1)
            plt.plot(f_f, pd_f / (np.max(pd_f)), label= sig + "_filtered", linewidth=1)
            plt.plot(0.1 * np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
            plt.xlim([-0.01, 0.2])
            plt.title('Power Spectral Density of ' + sig + ' in ' + dat[i][j].name)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Scaled PSD')
            plt.grid()
            plt.legend()
            plt.savefig("figs/tst_filt/" + dat[i][j].name + "/" + sig + "_psd.png", dpi=150)
            plt.close()

plt.close('all')