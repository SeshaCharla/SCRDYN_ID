import sys

sys.path.append('../ReadStuff/')

import read_data as rd
import matplotlib.pyplot as plt
import numpy as np
import psd

import matplotlib as mpl
mpl.use('TkAgg')


# Read data
tst = rd.load_test_data_set()
td = 0.2
fs = 1/td

# Welch PSD for x1
for sig in ['x1', 'x2', 'u1', 'u2', 'T', 'F']:
    plt.figure()
    for i in range(2):
        for j in range(3):
            f, pd = psd.welch_psd(tst[i][j].ssd[sig], fs)
            plt.plot(f, pd/(np.max(pd)), label=tst[i][j].name, linewidth=1)
            plt.plot(0.1*np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
    plt.xlim([-0.01, 0.2])
    plt.title('Power Spectral Density of '+ sig)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Scaled PSD')
    plt.grid()
    plt.legend()
    plt.savefig("figs/test_psd/"+sig+".png", dpi=1200)

plt.figure()
for i in range(2):
    for j in range(3):
        f, pd = psd.welch_psd(tst[i][j].iod['y1'], fs)
        plt.plot(f, pd / (np.max(pd)), label=tst[i][j].name, linewidth=1)
        plt.plot(0.1 * np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
plt.xlim([-0.01, 0.2])
plt.title('Power Spectral Density of '+ 'y1')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Scaled PSD')
plt.grid()
plt.legend()
plt.savefig("figs/test_psd/y1.png", dpi=1200)

plt.show()
