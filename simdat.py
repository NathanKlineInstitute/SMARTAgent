import numpy as np
from pylab import *
import pickle
import pandas as pd
from conf import dconf

ion()

simConfig = pickle.load(open('data/'+dconf['sim']['name']+'simConfig.pkl','rb'))

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

actreward = pd.DataFrame(np.loadtxt('data/'+dconf['sim']['name']+'ActionsRewards.txt'),columns=['time','action','reward']) 

#
def plotavgweights (pdf):
  utimes = np.unique(pdf.time)
  All_ML_weights = []
  All_MR_weights = []
  for t in utimes:
      for pop, arr in zip(['ML','MR'],[All_ML_weights,All_MR_weights]):
        pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[pop]) & (pdf.postid<=dendidx[pop]) & (pdf.preid>=dstartidx['V1']) & (pdf.preid<=dendidx['V1'])]
        arr.append(np.mean(pdfs.weight))
  subplot(2,1,1)
  plot(actreward.time,actreward.reward,'k',linewidth=4)
  plot(actreward.time,actreward.reward,'ko',markersize=10)  
  xlim((0,simConfig['simConfig']['duration']))
  ylim((-1.1,1.1))
  subplot(2,1,2)
  plot(utimes,All_ML_weights,'r-',linewidth=3);
  plot(utimes,All_MR_weights,'b-',linewidth=3); 
  legend(('V->ML','V->MR'),loc='upper left')
  plot(utimes,All_ML_weights,'ro',markersize=10);  plot(utimes,All_MR_weights,'bo',markersize=10)
  xlabel('Time (ms)'); ylabel('RL weights')
  xlim((0,simConfig['simConfig']['duration']))
  return All_ML_weights, All_MR_weights

All_ML_weights,All_MR_Weights = plotavgweights(pdf)
