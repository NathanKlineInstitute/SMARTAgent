import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

simConfig = pickle.load(open('data/simConfig.pkl','rb'))

dstartidx = {p:simConfig['net']['pops'][p]['cellGids'][0] for p in simConfig['net']['pops'].keys()} # starting indices for each population
dendidx = {p:simConfig['net']['pops'][p]['cellGids'][-1] for p in simConfig['net']['pops'].keys()} # ending indices for each population

#
def readinweights (d):
  A = []
  ddsyn = d['simData']['synweights']
  for rank in ddsyn.keys():
    dsyn = ddsyn[rank]
    for lsyn in dsyn:
      A.append(lsyn)
  return pd.DataFrame(A,columns=['time','stdptype','preid','postid','weight'])
  
pdf = readinweights(simConfig)

def plotavgweights (pdf):
  utimes = np.unique(pdf.time)
  #np.zeros(shape=(20,20))
  All_ML_weights = []
  All_MR_weights = []
  for t in utimes:
      for pop, arr in zip(['ML','MR'],[All_ML_weights,All_MR_weights]):
        pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[pop]) & (pdf.postid<=dendidx[pop])]
        arr.append(np.mean(pdfs.weight))
  #plt.subplot(1,2,1)
  plt.plot(utimes,All_ML_weights,'r-')
  plt.plot(utimes,All_MR_weights,'b-')
  plt.xlabel('time [msec]')
  plt.legend(('ML','MR'),loc='upper left')
  plt.title('Weights')
  plt.show()

plotavgweights(pdf)
