import sys

from sympy.printing.pretty.pretty_symbology import line_width

sys.path.append('../ReadStuff/')

import read_data as rd
import matplotlib.pyplot as plt
import numpy as np
import psd

import matplotlib as mpl
mpl.use('TkAgg')


# Read data
trk = rd.load_truck_data_set()
td = 1
fs = 1/td

# Welch PSD for x1
for sig in ['y1', 'u1', 'u2', 'T', 'F']:
    plt.figure()
    for i in range(2):
        for j in range(4):
            f, pd = psd.welch_psd(trk[i][j].iod[sig], fs)
            plt.plot(f, pd/(np.max(pd)), label=trk[i][j].name, linewidth=1)
            plt.plot(0.05*np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
    plt.xlim([-0.01, 0.2])
    plt.title('Power Spectral Density of '+ sig)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Scaled PSD')
    plt.grid()
    plt.legend()
    plt.savefig("figs/truck_psd/"+sig+".png", dpi=1200)

plt.show()
