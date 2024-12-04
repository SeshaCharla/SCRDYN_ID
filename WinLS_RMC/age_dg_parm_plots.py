import matplotlib.pyplot as plt
import pickle

import matplotlib
matplotlib.use('TkAgg')


bt_age = pickle.load(open('bt_age.pkl', 'rb'))
bt_dg = pickle.load(open('bt_dg.pkl', 'rb'))

for i in range(5):
    plt.figure()
    plt.plot(bt_age[:, i], 'x', label='theta_{} aged'.format(i+1))
    plt.plot(bt_dg[:, i], 'o', label='theta_{} dg'.format(i+1))
    plt.legend(loc='best')
    plt.grid()
plt.show()