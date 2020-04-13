import numpy as np
from pylab import *
import pickle
import pandas as pd
from conf import dconf
import os
import sys
import anim

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
  return pd.DataFrame(A,columns=['time','stdptype','preid','postid','weight','syntype'])

def loadsimdat (name=None):
  # load simulation data
  if name is None and stepNB is -1: name = dconf['sim']['name']
  elif name is None and stepNB > -1: name = dconf['sim']['name'] + '_step_' + str(stepNB) + '_'
  simConfig = pickle.load(open('data/'+name+'simConfig.pkl','rb'))
  dstartidx = {p:simConfig['net']['pops'][p]['cellGids'][0] for p in simConfig['net']['pops'].keys()} # starting indices for each population
  dendidx = {p:simConfig['net']['pops'][p]['cellGids'][-1] for p in simConfig['net']['pops'].keys()} # ending indices for each population
  pdf = readinweights(simConfig)
  actreward = pd.DataFrame(np.loadtxt('data/'+name+'ActionsRewards.txt'),columns=['time','action','reward','proposed','hit'])   
  return simConfig, pdf, actreward, dstartidx, dendidx

#
def plotSynWeightsPerTimeStep(pdf,pauset=1, gifpath=None, mp4path=None, framerate=5):
    utimes = np.unique(pdf.time)
    maxNMDAwt = np.max(pdf[pdf.syntype=='NMDA']).weight
    maxAMPAwt = np.max(pdf[pdf.syntype=='AMPA']).weight
    maxwt = maxNMDAwt+maxAMPAwt
    minNMDAwt = np.min(pdf[pdf.syntype=='NMDA']).weight
    minAMPAwt = np.min(pdf[pdf.syntype=='AMPA']).weight
    minwt = minNMDAwt+minAMPAwt
    wtrange = 0.1*(maxwt-minwt)
    fig = plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(4,8)
    f_ax1 = fig.add_subplot(gs[0,0:3])
    f_ax2 = fig.add_subplot(gs[0,4:7])
    pdfsL = pdf[(pdf.postid>=dstartidx['EML']) & (pdf.postid<=dendidx['EML'])]
    pdfsR = pdf[(pdf.postid>=dstartidx['EMR']) & (pdf.postid<=dendidx['EMR'])]
    Lwts = [] #wts of connections onto EML
    Rwts = [] #wts of connections onto EMR
    for t in utimes:
        ct_Lwts = pdfsL[(pdfsL.time==t)].weight
        ct_Rwts = pdfsR[(pdfsR.time==t)].weight
        Lwts.append(np.mean(ct_Lwts))
        Rwts.append(np.mean(ct_Rwts))
    actionvsproposed = actreward.action-actreward.proposed
    followtheball = actreward[actionvsproposed==0]
    f_ax1.plot(actreward.time,actreward.reward,'ko',markersize=2)
    #f_ax1.plot(actreward.time,actreward.hit,'go',markersize=4) #mark times when the racket hits the ball
    #f_ax1.plot(followtheball.time,followtheball.action-followtheball.proposed,'ko',markersize=4) #mark times when the racket follows the ball
    #if dconf['rewardcodes']['scorePoint']==1:
    #    scoretpnts = actreward.time[actreward.reward>dconf['rewardcodes']['scorePoint']]
    #    f_ax1.plot(scoretpnts,2*np.ones(shape=(len(scoretpnts),1)),'bo',markersize=4)  
    f_ax1.set_xlim((0,simConfig['simConfig']['duration']))
    f_ax1.set_ylim((np.min(actreward.reward),np.max(actreward.reward)))
    f_ax1.set_ylabel('rewards')
    f_ax1.set_xlabel('time (msec)')
    #plot mean weights of all connections onto EML and EMR
    f_ax2.plot(utimes,Lwts,'r-o',markersize=1)
    f_ax2.plot(utimes,Rwts,'b-o',markersize=1)
    f_ax2.set_xlim((0,simConfig['simConfig']['duration']))
    f_ax2.set_ylim((np.min([np.min(Lwts),np.min(Rwts)]),np.max([np.max(Lwts),np.max(Rwts)])))
    f_ax2.set_ylabel('total weights')
    f_ax2.set_xlabel('time (msec)')
    f_ax2.legend(('->EML','->EMR'),loc='upper left')
    #f_ax1.legend(('HitBall','FollowBall','ScorePoint'),loc='upper left')
    f_ax = []
    for rows in range(3):
        for cols in range(8):
            f_ax.append(fig.add_subplot(gs[rows+1,cols]))
    #fig, axs = plt.subplots(3, 8, figsize=(18,10)); lax = axs.ravel()
    cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2]) 
    ltitle = ['Excit V1->ML','Excit V1->MR', 'Excit V4->ML','Excit V4->MR', 'Excit MT->ML', 'Excit MT->MR','Excit DirE->ML','Excit DirE->MR','Excit DirNE->ML','Excit DirNE->MR','Excit DirN->ML','Excit DirN->MR','Excit DirNW->ML','Excit DirNW->MR','Excit DirW->ML','Excit DirW->MR','Excit DirSW->ML','Excit DirSW->MR','Excit DirS->ML','Excit DirS->MR','Excit DirSE->ML','Excit DirSE->MR']
    lfnimage = []
    tinds = 0
    maxtstr = len(str(len(utimes)))
    for t in utimes:
        f_ax1.plot([t,t],[np.min(actreward.reward),np.max(actreward.reward)],'r',linewidth=0.2)
        f_ax2.plot([t,t],[np.min([np.min(Lwts),np.min(Rwts)]),np.max([np.max(Lwts),np.max(Rwts)])],'r',linewidth=0.2)
        pinds = 0
        tinds = tinds+1
        fig.suptitle('Connection Weights at Time ' + str(t) + ' ms')
        for src in ['EV1', 'EV4', 'EMT','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE']:
            cpdfL = pdf[(pdf.time==t) & (pdf.postid>=dstartidx['EML']) & (pdf.postid<=dendidx['EML']) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
            wts1l = np.array(cpdfL.weight)
            wts2l = np.reshape(wts1l,(int(len(wts1l)/2),2))
            wtsl = np.sum(wts2l,1)
            wtsL = np.reshape(wtsl,(int(np.sqrt(len(wtsl))),int(np.sqrt(len(wtsl))))) #assuming neurons in each layer are in square configuration--may need adaptation later
            cpdfR = pdf[(pdf.time==t) & (pdf.postid>=dstartidx['EMR']) & (pdf.postid<=dendidx['EMR']) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
            wts1r = np.array(cpdfL.weight)
            wts2r = np.reshape(wts1r,(int(len(wts1r)/2),2))
            wtsr = np.sum(wts2r,1)
            wtsR = np.reshape(wtsr,(int(np.sqrt(len(wtsr))),int(np.sqrt(len(wtsr))))) #assuming neurons in each layer are in square configuration--may need adaptation later
            ax=f_ax[pinds]
            pcm = ax.imshow(wtsL, origin='upper', cmap='gray', vmin=minwt, vmax=maxwt+wtrange)
            ax.set_title(ltitle[pinds])
            pinds=pinds+1
            ax=f_ax[pinds]
            pcm = ax.imshow(wtsR, origin='upper', cmap='gray', vmin=minwt, vmax=maxwt+wtrange)
            ax.set_title(ltitle[pinds])
            if pinds==15: plt.colorbar(pcm, cax = cbaxes)
            pinds = pinds+1
        f_ax[22].axis('off')
        f_ax[23].axis('off')
        if gifpath is not None or mp4path is not None:
            ctstrl = len(str(tinds))
            tpre = ''
            for ttt in range(maxtstr-ctstrl):
                tpre = tpre+'0'
            fname = tpre+str(tinds)+'.png'
            fnimg = '/tmp/'+fname
            savefig(fnimg); lfnimage.append(fnimg)
        if pauset>0: plt.pause(pauset)
    if gifpath is not None: anim.savegif(lfnimage, gifpath)
    if mp4path is not None: anim.savemp4('/tmp/*.png', mp4path, framerate)
    for fn in lfnimage: os.unlink(fn) # remove the tmp files

        


def plotavgweights (pdf):
  utimes = np.unique(pdf.time)
  davgw = {}
  subplot(12,1,1)
  plot(actreward.time,actreward.reward,'k',linewidth=4)
  plot(actreward.time,actreward.reward,'ko',markersize=10)  
  xlim((0,simConfig['simConfig']['duration']))
  ylim((-1.1,1.1))
  gdx = 2
  for src in ['EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE', 'EV4', 'EMT']:
      for trg in ['EML', 'EMR']:
          davgw[src+'->'+trg] = arr = []        
          for t in utimes:
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
              arr.append(np.mean(pdfs.weight))
      subplot(12,1,gdx)
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
  for src in ['EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE', 'EV4', 'EMT']:
      figure(gdx)
      subplot(3,1,1)
      plot(actreward.time,actreward.reward,'k',linewidth=4)
      plot(actreward.time,actreward.reward,'ko',markersize=10)  
      xlim((0,simConfig['simConfig']['duration']))
      ylim((-1.1,1.1))
      ylabel('critic')
      title('sum of weights on to post-synaptic neurons')
      for trg in ['EML', 'EMR']:
          wperPostID[src+'->'+trg] = arr = []
          tstep = 0
          for t in utimes:
              arr.append([])
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
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
  return wperPostID

def plotavgweightsPerPostSynNeuron2(pdf):
  utimes = np.unique(pdf.time)
  #for every postsynaptic neuron, find total weight of synaptic inputs per area (i.e. synaptic inputs from EV1, EV4 and EIT and treated separately for each cell——if there are 200 unique cells, will get 600 weights as 200 from each originating layer)
  wperPostID = {}
  #gdx = 2   
  for src in ['EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE', 'EV4', 'EMT']:
      figure()
      subplot(3,1,1)
      plot(actreward.time,actreward.reward,'k',linewidth=4)
      plot(actreward.time,actreward.reward,'ko',markersize=10)  
      xlim((0,simConfig['simConfig']['duration']))
      ylim((-1.1,1.1))
      ylabel('critic')
      colorbar
      title('sum of weights on to post-synaptic neurons')
      for trg in ['EML', 'EMR']:
          wperPostID[src+'->'+trg] = arr = []
          tstep = 0
          for t in utimes:
              arr.append([])
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
              uniqueCells = np.unique(pdfs.postid)
              for cell in uniqueCells:
                  pdfs1 = pdfs[(pdfs.postid==cell)]
                  arr[tstep].append(np.mean(pdfs1.weight))
              tstep += 1
      subplot(3,1,2)
      imshow(np.transpose(np.array(wperPostID[src+'->EML'])),aspect = 'auto',cmap='hot', interpolation='None')
      b1 = gca().get_xticks()
      gca().set_xticks(b1-1)
      gca().set_xticklabels((100*b1).astype(int))
      colorbar(orientation='horizontal',fraction=0.05)
      #legend((src+'->EML'),loc='upper left')
      xlim((-1,b1[-1]-1))
      ylabel(src+'->EML weights')
      subplot(3,1,3)
      imshow(np.transpose(np.array(wperPostID[src+'->EMR'])),aspect = 'auto',cmap='hot', interpolation='None') 
      b2 = gca().get_xticks()
      gca().set_xticks(b2-1)
      gca().set_xticklabels((100*b2).astype(int))
      colorbar(orientation='horizontal',fraction=0.05)
      #legend((src+'->EMR'),loc='upper left')       
      xlim((-1,b2[-1]-1))
      ylabel(src+'->EMR weights') 
      xlabel('Time (ms)')
  
def plotIndividualSynWeights(pdf):
  #plot 10% randomly selected connections
  utimes = np.unique(pdf.time)
  #for every postsynaptic neuron, find total weight of synaptic inputs per area (i.e. synaptic inputs from EV1, EV4 and EIT and treated separately for each cell——if there are 200 unique cells, will get 600 weights as 200 from each originating layer)
  allweights = {}
  preNeuronIDs = {}
  postNeuronIDs = {}
  #gdx = 2   
  for src in ['EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE', 'EV4', 'EMT']:
      #figure()
      #subplot(9,1,1)
      #plot(actreward.time,actreward.reward,'k',linewidth=4)
      #plot(actreward.time,actreward.reward,'ko',markersize=10)  
      #xlim((0,simConfig['simConfig']['duration']))
      #ylim((-1.1,1.1))
      #ylabel('critic')
      #colorbar
      #title('sum of weights on to post-synaptic neurons')
      for trg in ['EML', 'EMR']:
          allweights[src+'->'+trg] = arr = []
          preNeuronIDs[src+'->'+trg] = arr2 = []
          postNeuronIDs[src+'->'+trg] = arr3 = []
          tstep = 0
          for t in utimes:
              arr.append([])
              arr2.append([])
              arr3.append([])
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
              uniqueCells = np.unique(pdfs.postid)
              for cell in np.sort(np.random.choice(uniqueCells,int(0.1*len(uniqueCells)))):
                  pdfs1 = pdfs[(pdfs.postid==cell)]
                  p1 = np.array(pdfs1.weight)
                  preIDs1 = np.array(pdfs1.preid) #ID of pre synaptic neuron
                  for w in p1:
                      arr[tstep].append(w)
                  for preID in preIDs1:
                      arr2[tstep].append(preID)
                      arr3[tstep].append(cell)
              tstep += 1
      #subplot(2,1,1)
      figure()
      subplot(position=[0.05,0.1,0.01,0.8])
      c1 = get_cmap('viridis',1028)
      imshow(np.transpose(np.array(preNeuronIDs[src+'->EML'])),aspect = 'auto',cmap=c1, interpolation='None')
      subplot(position=[0.15,0.1,0.8,0.8])
      imshow(np.transpose(np.array(allweights[src+'->EML'])),aspect = 'auto',cmap='hot', interpolation='None')
      b1 = gca().get_xticks()
      gca().set_xticks(b1-1)
      gca().set_xticklabels((100*b1).astype(int))
      colorbar(orientation='vertical',fraction=0.01)
      #legend((src+'->EML'),loc='upper left')
      xlim((-1,b1[-1]-1))
      ylabel(src+'->EML weights')
      xlabel('Time (ms)')
      subplot(position=[0.98,0.1,0.01,0.8])
      imshow(np.transpose(np.array(postNeuronIDs[src+'->EML'])),aspect = 'auto',cmap=c1, interpolation='None')
      #subplot(2,1,2)
      figure()
      subplot(position=[0.05,0.1,0.01,0.8])
      imshow(np.transpose(np.array(preNeuronIDs[src+'->EMR'])),aspect = 'auto',cmap=c1, interpolation='None')
      subplot(position=[0.15,0.1,0.8,0.8])
      imshow(np.transpose(np.array(allweights[src+'->EMR'])),aspect = 'auto',cmap='hot', interpolation='None') 
      b2 = gca().get_xticks()
      gca().set_xticks(b2-1)
      gca().set_xticklabels((100*b2).astype(int))
      colorbar(orientation='vertical',fraction=0.01)
      #legend((src+'->EMR'),loc='upper left')       
      xlim((-1,b2[-1]-1))
      ylabel(src+'->EMR weights') 
      xlabel('Time (ms)')
      subplot(position=[0.98,0.1,0.01,0.8])
      imshow(np.transpose(np.array(postNeuronIDs[src+'->EMR'])),aspect = 'auto',cmap=c1, interpolation='None')

def plotSynWeightsPostNeuronID(pdf,postNeuronID):
  utimes = np.unique(pdf.time)
  #for a postID, find a neuron in ML and a neuron in MR
  pdfs_ML = pdf[(pdf.time==utimes[0]) & (pdf.postid>=dstartidx['EML']) & (pdf.postid<=dendidx['EML'])]
  uIDs_ML = np.unique(pdfs_ML.postid)
  pdfs_MR = pdf[(pdf.time==utimes[0]) & (pdf.postid>=dstartidx['EMR']) & (pdf.postid<=dendidx['EMR'])]
  uIDs_MR = np.unique(pdfs_MR.postid)
  targetML_postID = min(uIDs_ML)-1+postNeuronID
  targetMR_postID = min(uIDs_MR)-1+postNeuronID

  NBpreN_ML = len(np.unique(pdfs_ML.preid))
  NBpreN_MR = len(np.unique(pdfs_MR.preid)) 
  #for every postsynaptic neuron, find total weight of synaptic inputs per area (i.e. synaptic inputs from EV1, EV4 and EIT and treated separately for each cell——if there are 200 unique cells, will get 600 weights as 200 from each originating layer)
  MLweights = {}
  MRweights = {}
  MLpreNeuronIDs = {}
  MRpreNeuronIDs = {}
  
  #for each of those neurons, find presynaptic neuron IDs and the strengths
  #gdx = 2
  figure()
  subplot(12,1,1)
  plot(actreward.time,actreward.reward,'k',linewidth=4)
  plot(actreward.time,actreward.reward,'ko',markersize=10)  
  xlim((0,simConfig['simConfig']['duration']))
  #ylim((-1.1,1.1))
  ylim((min(actreward.reward),max(actreward.reward)))
  ylabel('critic')
  title('weights of all connections for a post-synaptic neuron')
  pdx = 2    
  for src in ['EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE', 'EV4', 'EMT']:
      MLweights[src] = arrL = []
      MRweights[src] = arrR = []
      MLpreNeuronIDs[src] = arrL2 = []
      MRpreNeuronIDs[src] = arrR2 = []
      tstep = 0
      for t in utimes:
          arrL.append([])
          arrR.append([])
          arrL2.append([])
          arrR2.append([])
          pdfsL = pdf[(pdf.time==t) & (pdf.postid==targetML_postID) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
          pdfsR = pdf[(pdf.time==t) & (pdf.postid==targetMR_postID) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
          upreLCells = np.unique(pdfsL.preid)
          upreRCells = np.unique(pdfsR.preid)
          for preID in upreLCells:
              pdfs1 = pdfsL[(pdfsL.preid==preID)]
              p1 = np.array(pdfs1.weight) #may have more than 1 weight---as two cells may have both AMPA and NMDA syns
              for w in p1:
                  arrL[tstep].append(w)
                  arrL2[tstep].append(preID)
          for preID in upreRCells:
              pdfs2 = pdfsR[(pdfsR.preid==preID)]
              p2 = np.array(pdfs2.weight) #may have more than 1 weight---as two cells may have both AMPA and NMDA syns
              for w in p2:
                  arrR[tstep].append(w)
                  arrR2[tstep].append(preID)
          tstep += 1
      subplot(12,1,pdx)
      plot(utimes,np.array(MLweights[src]),'r-o',linewidth=3,markersize=5)
      plot(utimes,np.array(MRweights[src]),'b-o',linewidth=3,markersize=5)
      legend((src+'->EML',src+'->EMR'),loc='upper left')
      xlim((0,simConfig['simConfig']['duration']))
      pdx += 1
  
      

if __name__ == '__main__':
  if sys.argv[1] is -1:
    stepNB = -1
  else:
    stepNB = int(sys.argv[1]) #which file(stepNB) want to plot
  print(stepNB)
  simConfig, pdf, actreward, dstartidx, dendidx = loadsimdat()
  #davgw = plotavgweights(pdf)
  plotSynWeightsPerTimeStep(pdf,pauset=1,mp4path='data/'+dconf['sim']['name']+'weightmap.mp4', framerate=10) #plot images
  #wperPostID = plotavgweightsPerPostSynNeuron1(pdf)
  #plotavgweightsPerPostSynNeuron2(pdf)
  #plotIndividualSynWeights(pdf)
  #plotSynWeightsPostNeuronID(pdf,5)
  #plotSynWeightsPostNeuronID(pdf,15)
  #plotSynWeightsPostNeuronID(pdf,25)
  #plotSynWeightsPostNeuronID(pdf,35)
  #plotSynWeightsPostNeuronID(pdf,45)
