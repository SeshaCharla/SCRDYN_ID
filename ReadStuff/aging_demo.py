import read_data as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')


tst_data = rd.load_test_data_set()


for tst in range(3):
    plt.figure(3*tst)
    plt.title("NOx reduced at time-step k")
    plt.figure(3*tst+1)
    plt.title("Urea Injection")
    plt.figure(3*tst+2)
    plt.title("Ammonia Slip")
    for age in range(2):
        dat = tst_data[age][tst]
        plt.figure(3*tst)
        plt.plot(dat.ssd['t'], dat.ssd['eta'], label=dat.name+': u1(k) - x1(k+1)')
        plt.figure(3*tst+1)
        plt.plot(dat.ssd['t'], dat.ssd['u2'], label=dat.name+': u_inj')
        plt.figure(3*tst+2)
        plt.plot(dat.ssd['t'], dat.ssd['x2'], label=dat.name+': x2(k)')

for i in range(9):
    plt.figure(i)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Time')
    plt.savefig("./aging_demo/"+str(i)+".png")
plt.show()
