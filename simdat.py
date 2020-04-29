import numpy as np
from pylab import *
import pickle
import pandas as pd
from conf import dconf
import os
import sys
import anim
from matplotlib import animation

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
  print('loading data from', name)
  simConfig = pickle.load(open('data/'+name+'simConfig.pkl','rb'))
  dstartidx = {p:simConfig['net']['pops'][p]['cellGids'][0] for p in simConfig['net']['pops'].keys()} # starting indices for each population
  dendidx = {p:simConfig['net']['pops'][p]['cellGids'][-1] for p in simConfig['net']['pops'].keys()} # ending indices for each population
  pdf = readinweights(simConfig)
  actreward = pd.DataFrame(np.loadtxt('data/'+name+'ActionsRewards.txt'),columns=['time','action','reward','proposed','hit'])
  dnumc = {p:dendidx[p]-dstartidx[p]+1 for p in simConfig['net']['pops'].keys()}
  return simConfig, pdf, actreward, dstartidx, dendidx, dnumc

def loadInputImages (fn):
  print('loading input images from', fn)
  Input_Images = np.loadtxt(fn)
  New_InputImages = []
  NB_Images = int(Input_Images.shape[0]/Input_Images.shape[1])
  for x in range(NB_Images):
    fp = x*Input_Images.shape[1]
    cImage = Input_Images[fp:fp+20,:] # 20 is sqrt of 400 (20x20 pixels)
    New_InputImages.append(cImage)
  New_InputImages = np.array(New_InputImages)
  return New_InputImages

#
def animSynWeights (pdf, outpath, framerate=10, figsize=(7,4), cmap='jet'):
  # animate the synaptic weights along with some stats on behavior
  origfsz = rcParams['font.size']; rcParams['font.size'] = 5; ioff() # save original font size, turn off interactive plotting
  utimes = np.unique(pdf.time)
  #maxNMDAwt = np.max(pdf[pdf.syntype=='NMDA'].weight)
  #maxAMPAwt = np.max(pdf[pdf.syntype=='AMPA'].weight)
  #maxwt = max(amax(maxNMDAwt),amax(maxAMPAwt))
  #maxwt = maxNMDAwt+maxAMPAwt
  #minNMDAwt = np.min(pdf[pdf.syntype=='NMDA'].weight)
  #minAMPAwt = np.min(pdf[pdf.syntype=='AMPA'].weight)
  #minwt = min(amin(minNMDAwt),amin(minAMPAwt))
  #minwt = minNMDAwt+minAMPAwt
  minwt = np.min(pdf.weight)
  maxwt = np.max(pdf.weight)
  wtrange = 0.1*(maxwt-minwt)
  print('minwt:',minwt,'maxwt:',maxwt)
  if figsize is not None: fig = plt.figure(figsize=figsize)
  else: fig = plt.figure()
  gs = fig.add_gridspec(4,8)
  f_ax = []
  ax_count = 0
  for rows in range(3):
    for cols in range(8): 
      if ax_count<22: 
        f_ax.append(fig.add_subplot(gs[rows,cols]))
      ax_count += 1
  cbaxes = fig.add_axes([0.92, 0.4, 0.01, 0.2])
  f_ax1 = fig.add_subplot(gs[2,6:8])
  f_ax2 = fig.add_subplot(gs[3,0:2])
  f_ax3 = fig.add_subplot(gs[3,3:5])
  f_ax4 = fig.add_subplot(gs[3,6:8])
  pdfsL = pdf[(pdf.postid>=dstartidx['EML']) & (pdf.postid<=dendidx['EML'])]
  pdfsR = pdf[(pdf.postid>=dstartidx['EMR']) & (pdf.postid<=dendidx['EMR'])]
  Lwts = [np.mean(pdfsL[(pdfsL.time==t)].weight) for t in utimes] #wts of connections onto EML
  Rwts = [np.mean(pdfsR[(pdfsR.time==t)].weight) for t in utimes] #wts of connections onto EMR
  action_times = np.array(actreward.time)
  actionvsproposed = np.array(actreward.action-actreward.proposed)
  rewardingActions = np.cumsum(np.where(actionvsproposed==0,1,0)) #rewarding action
  #punishing action i.e. when the action leads to move the racket away from the ball
  punishingActions = np.cumsum(np.where((actionvsproposed>0) | (actionvsproposed<0),1,0)) 
  cumActs = np.array(range(1,len(actionvsproposed)+1))
  Hit_Missed = np.array(actreward.hit)
  allHit = np.where(Hit_Missed==1,1,0) 
  allMissed = np.where(Hit_Missed==-1,1,0)
  cumHits = np.cumsum(allHit) #cumulative hits evolving with time.
  cumMissed = np.cumsum(allMissed) #if a reward is -1, replace it with 1 else replace it with 0.
  f_ax1.plot(action_times,np.divide(rewardingActions,cumActs),'r.',markersize=1)
  f_ax1.plot(action_times,np.divide(punishingActions,cumActs),'b.',markersize=1)
  f_ax1.set_xlim((0,np.max(action_times)))
  f_ax1.set_ylim((0,1))
  f_ax1.legend(('Follow Ball','Not Follow'),loc='upper left')
  f_ax2.plot(actreward.time,actreward.reward,'ko-',markersize=1)
  f_ax2.set_xlim((0,simConfig['simConfig']['duration']))
  f_ax2.set_ylim((np.min(actreward.reward),np.max(actreward.reward)))
  f_ax2.set_ylabel('Rewards'); #f_ax1.set_xlabel('Time (ms)')
  #plot mean weights of all connections onto EML and EMR
  f_ax3.plot(utimes,Lwts,'r-o',markersize=1)
  f_ax3.plot(utimes,Rwts,'b-o',markersize=1)
  f_ax3.set_xlim((0,simConfig['simConfig']['duration']))
  f_ax3.set_ylim((np.min([np.min(Lwts),np.min(Rwts)]),np.max([np.max(Lwts),np.max(Rwts)])))
  f_ax3.set_ylabel('Average weight'); #f_ax2.set_xlabel('Time (ms)')
  f_ax3.legend(('->EML','->EMR'),loc='upper left')
  f_ax4.plot(action_times,cumHits,'g-o',markersize=1)
  f_ax4.plot(action_times,cumMissed,'k-o',markersize=1)
  f_ax4.set_xlim((0,np.max(action_times)))
  f_ax4.set_ylim((0,np.max([cumHits[-1],cumMissed[-1]])))
  f_ax4.legend(('Hit Ball','Miss Ball'),loc='upper left')
  lsrc = ['EV1', 'EV4', 'EMT','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE']
  ltitle = []
  for src in lsrc:
    for trg in ['ML', 'MR']: ltitle.append(src+'->'+trg)
  dimg = {}; dline = {}; 
  def getwts (tdx, src):
    t = utimes[tdx]
    ltarg = ['EML', 'EMR']
    lout = []
    for targ in ltarg:
      cpdf = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[targ]) & (pdf.postid<=dendidx[targ]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
      wts = np.array(cpdf.weight)
      sz = int(ceil(np.sqrt(len(wts))))
      sz2 = sz*sz
      #print(targ,len(wts),sz,sz2)
      lwt = [x for x in wts]
      while len(lwt) < sz2: lwt.append(lwt[-1])
      lwt=np.array(lwt)
      lwt = np.reshape(lwt, (sz,sz))
      lout.append(lwt)
    return lout[0], lout[1]    
  minR,maxR = np.min(actreward.reward),np.max(actreward.reward)
  minW,maxW = np.min([np.min(Lwts),np.min(Rwts)]), np.max([np.max(Lwts),np.max(Rwts)])
  t = utimes[0]
  dline[1], = f_ax1.plot([t,t],[minR,maxR],'r',linewidth=0.2); f_ax1.set_xticks([])
  dline[2], = f_ax2.plot([t,t],[minW,maxW],'r',linewidth=0.2); f_ax2.set_xticks([])  
  pinds = 0
  fig.suptitle('Time=' + str(round(t,2)) + ' ms')
  for src in lsrc:
    wtsL, wtsR = getwts(0, src)
    ax=f_ax[pinds]
    dimg[pinds] = ax.imshow(wtsL, origin='upper', cmap=cmap, vmin=minwt, vmax=maxwt+wtrange)
    ax.set_title(ltitle[pinds]); ax.axis('off')
    pinds+=1
    ax=f_ax[pinds]
    dimg[pinds] = ax.imshow(wtsR, origin='upper', cmap=cmap, vmin=minwt, vmax=maxwt+wtrange)
    ax.set_title(ltitle[pinds]); ax.axis('off')
    if pinds==15: plt.colorbar(dimg[pinds], cax = cbaxes)
    pinds+=1
  def updatefig (tdx):
    t = utimes[tdx]
    print('frame t = ', str(round(t,2)))
    dline[1].set_data([t,t],[minR,maxR])
    dline[2].set_data([t,t],[minW,maxW])
    pinds = 0
    fig.suptitle('Time=' + str(round(t,2)) + ' ms')
    for src in lsrc:
      wtsL, wtsR = getwts(tdx, src)
      dimg[pinds].set_data(wtsL)
      pinds+=1
      dimg[pinds].set_data(wtsR)
      pinds+=1
    return fig
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=len(utimes))
  writer = anim.getwriter(outpath, framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  rcParams['font.size'] = origfsz; ion() # restore original font size, restore interactive plotting
        
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
  print('loaded simulation data')
  #davgw = plotavgweights(pdf)
  animSynWeights(pdf[pdf.syntype=='AMPA'],'data/'+dconf['sim']['name']+'_AMPA_weightmap.mp4', framerate=10) #plot/save images as movie
  animSynWeights(pdf[pdf.syntype=='NMDA'],'data/'+dconf['sim']['name']+'_NMDA_weightmap.mp4', framerate=10) #plot/save images as movie  
  #wperPostID = plotavgweightsPerPostSynNeuron1(pdf)
  #plotavgweightsPerPostSynNeuron2(pdf)
  #plotIndividualSynWeights(pdf)
  #plotSynWeightsPostNeuronID(pdf,5)
  #plotSynWeightsPostNeuronID(pdf,15)
  #plotSynWeightsPostNeuronID(pdf,25)
  #plotSynWeightsPostNeuronID(pdf,35)
  #plotSynWeightsPostNeuronID(pdf,45)
