import sys
sys.path.append('../ReadStuff/')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import pickle
import test_parm_id as id
import test_parm_sim as sim
import read_data as rd

dat = rd.load_test_data_set()
parm = list([list(), list()])
parm_sim = list([list(), list()])

for age in range(2):
    for tst in range(3):
        data_set = dat[age][tst]
        print(data_set.ssd['t'])
        parm[age].append(id.tst_parm_id(data_set))

