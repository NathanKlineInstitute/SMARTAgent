import numpy as np
from pylab import *
import pickle
import pandas as pd
from conf import dconf
import os
import sys

ion()


global stepNB
stepNB = -1
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
  if name is None and stepNB is None: name = dconf['sim']['name']
  elif name is None and stepNB > -1: name = dconf['sim']['name'] + '_step_' + str(stepNB) + '_'
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

def plotavgweightsPerPostSynNeuron1(pdf):
  utimes = np.unique(pdf.time)
  #for every postsynaptic neuron, find total weight of synaptic inputs per area (i.e. synaptic inputs from EV1, EV4 and EIT and treated separately for each cell——if there are 200 unique cells, will get 600 weights as 200 from each originating layer)
  wperPostID = {}
  gdx = 2   
  for src in ['EV1', 'EV4', 'EIT']:
      figure(gdx)
      subplot(3,1,1)
      plot(actreward.time,actreward.reward,'k',linewidth=4)
      plot(actreward.time,actreward.reward,'ko',markersize=10)  
      xlim((0,simConfig['simConfig']['duration']))
      ylim((-1.1,1.1))
      for trg in ['EML', 'EMR']:
          wperPostID[src+'->'+trg] = arr = []
          tstep = 0
          for t in utimes:
              arr.append([])
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[trg])]
              uniqueCells = np.unique(pdfs.postid)
              for cell in uniqueCells:
                  pdfs1 = pdfs[(pdfs.postid==cell)]
                  arr[tstep].append(np.mean(pdfs1.weight))
              tstep += 1
      subplot(3,1,2)
      plot(utimes,np.array(wperPostID[src+'->EML']),'r-o',linewidth=3,markersize=5)
      #legend((src+'->EML'),loc='upper left')
      xlim((0,simConfig['simConfig']['duration']))
      ylabel(src+'->EML weights')
      subplot(3,1,3)
      plot(utimes,np.array(wperPostID[src+'->EMR']),'b-o',linewidth=3,markersize=5) 
      #legend((src+'->EMR'),loc='upper left')       
      xlim((0,simConfig['simConfig']['duration']))
      ylabel(src+'->EMR weights') 
      gdx += 1
      xlabel('Time (ms)')
      title('sum of weights on to post-synaptic neurons') 
  return wperPostID

def plotavgweightsPerPostSynNeuron2(pdf):
  utimes = np.unique(pdf.time)
  #for every postsynaptic neuron, find total weight of synaptic inputs per area (i.e. synaptic inputs from EV1, EV4 and EIT and treated separately for each cell——if there are 200 unique cells, will get 600 weights as 200 from each originating layer)
  wperPostID = {}
  #gdx = 2   
  for src in ['EV1', 'EV4', 'EIT']:
      figure()
      subplot(3,1,1)
      plot(actreward.time,actreward.reward,'k',linewidth=4)
      plot(actreward.time,actreward.reward,'ko',markersize=10)  
      xlim((0,simConfig['simConfig']['duration']))
      ylim((-1.1,1.1))
      for trg in ['EML', 'EMR']:
          wperPostID[src+'->'+trg] = arr = []
          tstep = 0
          for t in utimes:
              arr.append([])
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[trg])]
              uniqueCells = np.unique(pdfs.postid)
              for cell in uniqueCells:
                  pdfs1 = pdfs[(pdfs.postid==cell)]
                  arr[tstep].append(np.mean(pdfs1.weight))
              tstep += 1
      subplot(3,1,2)
      imshow(np.transpose(np.array(wperPostID[src+'->EML'])),aspect = 'auto', extent = (0.1,0.8,0.1,0.8), interpolation='None')
      b1 = gca().get_xticks()
      b1 = 1000*b1
      b1.astype(int)
      b1
      gca().set_xticklabels(b1)
      #legend((src+'->EML'),loc='upper left')
      #xlim((0,simConfig['simConfig']['duration']))
      ylabel(src+'->EML weights')
      subplot(3,1,3)
      imshow(np.transpose(np.array(wperPostID[src+'->EMR'])),aspect = 'auto', extent = (0.1,0.8,0.1,0.8), interpolation='None') 
      b2 = gca().get_xticks()
      b2 = 1000*b2
      b2.astype(int)
      b2
      gca().set_xticklabels(b2)
      #legend((src+'->EMR'),loc='upper left')       
      #xlim((0,simConfig['simConfig']['duration']))
      ylabel(src+'->EMR weights') 
      xlabel('Time (ms)')
      title('sum of weights on to post-synaptic neurons') 
  
if __name__ == '__main__':
  stepNB = int(sys.argv[1]) #which file(stepNB) want to plot
  if stepNB is None: stepNB = -1
  print(stepNB)
  simConfig, pdf, actreward, dstartidx, dendidx = loadsimdat()
  davgw = plotavgweights(pdf)
  #wperPostID = plotavgweightsPerPostSynNeuron1(pdf)
  plotavgweightsPerPostSynNeuron2(pdf)
