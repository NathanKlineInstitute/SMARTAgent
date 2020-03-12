import numpy as np
from pylab import *
import pickle
import pandas as pd
from conf import dconf
import os
import sys

ion()



#
def readinweights (d):
  # read the synaptic plasticity weights into a pandas dataframe
  A = []
  ddsyn = d['simData']['synweights']
  for rank in ddsyn.keys():
    dsyn = ddsyn[rank]
    for lsyn in dsyn:
      A.append(lsyn)
  return pd.DataFrame(A,columns=['time','stdptype','preid','postid','weight'])

def loadsimdat (name=None):
  # load simulation data
  stepNB = sys.argv[0] #which file(stepNB) want to plot
  print(stepNB)
  if stepNB is None: stepNB = -1
  if name is None and stepNB is None: name = dconf['sim']['name']
  elif name is None and stepNB > -1: name = dconf['sim']['name'] + '_step_' + str(stepNB)
  simConfig = pickle.load(open('data/'+name+'simConfig.pkl','rb'))
  dstartidx = {p:simConfig['net']['pops'][p]['cellGids'][0] for p in simConfig['net']['pops'].keys()} # starting indices for each population
  dendidx = {p:simConfig['net']['pops'][p]['cellGids'][-1] for p in simConfig['net']['pops'].keys()} # ending indices for each population
  pdf = readinweights(simConfig)
  actreward = pd.DataFrame(np.loadtxt('data/'+name+'ActionsRewards.txt'),columns=['time','action','reward','proposed'])   
  return simConfig, pdf, actreward, dstartidx, dendidx

#
def plotavgweights (pdf):
  utimes = np.unique(pdf.time)
  davgw = {}
  subplot(4,1,1)
  plot(actreward.time,actreward.reward,'k',linewidth=4)
  plot(actreward.time,actreward.reward,'ko',markersize=10)  
  xlim((0,simConfig['simConfig']['duration']))
  ylim((-1.1,1.1))
  gdx = 2
  for src in ['EV1', 'EV4', 'EIT']:
      for trg in ['EML', 'EMR']:
          davgw[src+'->'+trg] = arr = []        
          for t in utimes:
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[trg])]
              arr.append(np.mean(pdfs.weight))
      subplot(4,1,gdx)
      plot(utimes,davgw[src+'->EML'],'r-',linewidth=3);
      plot(utimes,davgw[src+'->EMR'],'b-',linewidth=3); 
      legend((src+'->EML',src+'->EMR'),loc='upper left')
      plot(utimes,davgw[src+'->EML'],'ro',markersize=10);
      plot(utimes,davgw[src+'->EMR'],'bo',markersize=10);       
      xlim((0,simConfig['simConfig']['duration']))
      ylabel('RL weights') 
      gdx += 1
  xlabel('Time (ms)'); 
  return davgw

if __name__ == '__main__':
  simConfig, pdf, actreward, dstartidx, dendidx = loadsimdat()
  davgw = plotavgweights(pdf)
