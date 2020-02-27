import numpy as np
from pylab import *
import pickle
import pandas as pd

ion()

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

actreward = np.loadtxt('data/ActionsRewards.txt') 

#
def plotavgweights (pdf):
  utimes = np.unique(pdf.time)
  All_ML_weights = []
  All_MR_weights = []
  for t in utimes:
      for pop, arr in zip(['ML','MR'],[All_ML_weights,All_MR_weights]):
        pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[pop]) & (pdf.postid<=dendidx[pop])]
        arr.append(np.mean(pdfs.weight))
  subplot(2,1,1)
  plot(actreward[:,0],actreward[:,2])
  xlim((0,simConfig['simConfig']['duration']))
  ylim((-1.1,1.1))
  subplot(2,1,2)
  plot(utimes,All_ML_weights,'r-')
  plot(utimes,All_MR_weights,'b-')
  xlabel('Time (ms)')
  legend(('ML','MR'),loc='upper left')
  ylabel('RL weights')
  xlim((0,simConfig['simConfig']['duration']))

plotavgweights(pdf)
