import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
simConfig = pickle.load(open('data/simConfig.pkl','rb'))

dstartidx = {p:simConfig['net']['pops'][p]['cellGids'][0] for p in simConfig['net']['pops'].keys()} # starting indices for each population
dendidx = {p:simConfig['net']['pops'][p]['cellGids'][-1] for p in simConfig['net']['pops'].keys()} # ending indices for each population

A = np.loadtxt('data/AdjustableWeights.txt')
postN = A[:,2]
postN_MR = np.where(postN>=dstartidx['MR'])
postN_ML = np.where(postN<dstartidx['MR'])#first one is bad
Mneurons_R = np.unique(postN[postN_MR])
Mneurons_L = np.unique(postN[postN_ML])

times = A[:,0]
utimes = np.unique(times)

#np.zeros(shape=(20,20))
All_ML_weights = []
All_MR_weights = []
for t in utimes:
    ctime_inds = np.where(times==t)
    ctime_postNs = A[ctime_inds,2]
    ctime_preNs = A[ctime_inds,1]
    ctime_weights = A[ctime_inds,4]
    ML_weights = []
    MR_weights = []
    for n in ctime_postNs[0]:
        cn_preNs = ctime_preNs[0][np.where(ctime_postNs[0]==n)]
        if len(cn_preNs)>len(np.unique(cn_preNs)):
            print(len(cn_preNs)-len(np.unique(cn_preNs)))
        cn_weights = ctime_weights[0][np.where(ctime_postNs[0]==n)]       
        if n in Mneurons_R:
            MR_weights.append(np.sum(cn_weights))
        elif n in Mneurons_L:
            ML_weights.append(np.sum(cn_weights))
    All_ML_weights.append(np.sum(ML_weights))
    All_MR_weights.append(np.sum(MR_weights))

#plt.subplot(1,2,1)
plt.plot(utimes,All_ML_weights,'r-')
plt.plot(utimes,All_MR_weights,'b-')
plt.xlabel('time [msec]')
plt.legend(('ML','MR'),loc='upper left')
plt.title('Weights')
plt.show()
