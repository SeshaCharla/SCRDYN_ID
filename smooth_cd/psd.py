import numpy as np
from scipy.signal import welch

def welch_psd(y, fs):
    """Wrapper around scipy.signal.welch
    With default parameters
    """
    seg_length = 2**8
    nf = 2**4
    f, pd = welch(y,
                  fs=fs,
                  window='bartlett',
                  nperseg=seg_length,
                  noverlap=None,
                  nfft=nf*seg_length,
                  detrend='linear',
                  return_onesided=True,
                  scaling='spectrum',
                  axis=-1,
                  average='mean')
    return f, pd

def welch_TD(y, t_skips, fs):
    """ Welch PSD with time-discontinuities
    """
    n = len(t_skips)-1
    psd = list()
    f = list()
    for i in range(n):
        if len(y[t_skips[i]:t_skips[i+1]]) > 0:
            f_i, psd_i = welch_psd(y[t_skips[i]:t_skips[i+1]], fs)
            f.append(f_i)
            psd.append(psd_i)
    return f, psd



if __name__ == "__main__":
    import sys
    sys.path.append('../ReadStuff/')
    import read_data as rd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('TkAgg')

    # Read data
    data = rd.Data("truck", 0, 1)
    f, pd = welch_psd(data.iod['y1'],  1)           # data.iod['t_skips'],
    # max_pd = np.max([np.max(pd_i) for pd_i in pd])
    plt.figure()
    plt.plot(f, pd)
    # for i in range(len(f)):
    plt.grid()
    plt.show()
