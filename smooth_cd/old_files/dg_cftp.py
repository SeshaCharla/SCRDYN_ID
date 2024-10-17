import sys

sys.path.append('../ReadStuff/')
import read_data as rd

import matplotlib.pyplot as plt
import numpy as np
import psd
import cdRLS_smoothing as cdRLS

import matplotlib as mpl
mpl.use('TkAgg')



dat = rd.Data('test', 0, 0)
lda = 0.98
f_samp = 5
h = dict()
nu = dict()
# Values
#
nu['x1'] = 0.5
nu['x2'] = 0.04
nu['u1'] = 0.5
nu['u2'] = 0.1
nu['T'] = 30
nu['F'] = 20
#
h['x1'] = 1
h['x2'] = 0.04
h['u1'] = 1
h['u2'] = 0.2
h['T'] = 40
h['F'] = 50

t = dat.ssd['t']
for y in ['x1', 'x2', 'u1', 'u2', 'T','F']: #
    (ys, g1, g2) = cdRLS.cdRLS_smooth(dat.ssd[y], lmda=lda, nu=nu[y], h=h[y])
    plt.figure()
    plt.plot(t, dat.ssd[y], label=y)
    plt.plot(t, ys, label=y+'_filtered')
    plt.plot(t, g1, label='g1')
    plt.plot(t, g2, label='g2')
    plt.legend()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel(y+'(mol/m^3)')
    plt.title(dat.name)
    plt.tight_layout()
    # plt.savefig("figs/" + dat.name + "/" + dat.name + "_y1.png", dpi=fig_dpi)

    plt.figure()
    f, pd = psd.welch_psd(dat.ssd[y], f_samp)
    fs, pds = psd.welch_psd(ys, f_samp)
    plt.plot(f, pd / (np.max(pd)), label=y, linewidth=1)
    plt.plot(fs, pds/(np.max(pds)), label=y+"_filtered", linewidth=1)
    plt.plot(0.1 * np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
    plt.xlim([-0.01, 0.2])
    plt.title('Power Spectral Density of '+y+' in '+dat.name)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Scaled PSD')
    plt.grid()
    plt.legend()
    # plt.savefig("figs/test_psd/" + sig + ".png", dpi=1200)


plt.show()

