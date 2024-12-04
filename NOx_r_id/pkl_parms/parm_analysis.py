import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')


f_names = ['dg_cftp.pkl', 'dg_hftp.pkl', 'dg_rmc.pkl',
           'aged_cftp.pkl', 'aged_hftp.pkl', 'aged_rmc.pkl',]
par_list = [pickle.load(open(f, 'rb')) for f in f_names]
dg_par = par_list[0:3]
ag_par = par_list[3:6]

for j in range(10):
    plt.figure()
    for i in range(3):
        plt.plot((dg_par[i])[j], 'bo')
        plt.plot((ag_par[i])[j], 'rx')
    plt.title(r'theta_'+str(j))
    plt.legend(['degreened', 'aged'])
    plt.savefig('../figs/theta_'+str(j)+'.png')
plt.show()
plt.close('all')
