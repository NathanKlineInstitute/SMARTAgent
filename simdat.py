import numpy as np
from pylab import *
import pickle
import pandas as pd
from conf import dconf
import os
import sys
import anim
from matplotlib import animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

ion()

global stepNB
stepNB = -1

def readweightsfile2pdf (fn):
  # read the synaptic plasticity weights saved as a dictionary into a pandas dataframe  
  D = pickle.load(open(fn,'rb'))
  A = []
  for preID in D.keys():
    for poID in D[preID].keys():
      for row in D[preID][poID]:
        A.append([row[0], preID, poID, syn, row[1]])
  return pd.DataFrame(A,columns=['time','preid','postid','weight'])

#
def readinweights (name):
  # read the synaptic plasticity weights associated with sim name into a pandas dataframe
  return readweightsfile2pdf('data/'+name+'synWeights.pkl')

def getsimname (name=None):
  if name is None:
    if stepNB is -1: name = dconf['sim']['name']
    elif stepNB > -1: name = dconf['sim']['name'] + '_step_' + str(stepNB) + '_'
  return name
  
def loadsimdat (name=None):
  # load simulation data
  name = getsimname(name)
  print('loading data from', name)
  simConfig = pickle.load(open('data/'+name+'simConfig.pkl','rb'))
  dstartidx = {p:simConfig['net']['pops'][p]['cellGids'][0] for p in simConfig['net']['pops'].keys()} # starting indices for each population
  dendidx = {p:simConfig['net']['pops'][p]['cellGids'][-1] for p in simConfig['net']['pops'].keys()} # ending indices for each population
  pdf = readinweights(name)
  actreward = pd.DataFrame(np.loadtxt('data/'+name+'ActionsRewards.txt'),columns=['time','action','reward','proposed','hit'])
  dnumc = {p:dendidx[p]-dstartidx[p]+1 for p in simConfig['net']['pops'].keys()}
  spkID= np.array(simConfig['simData']['spkid'])
  spkT = np.array(simConfig['simData']['spkt'])
  dspkID,dspkT = {},{}
  for pop in simConfig['net']['pops'].keys():
    dspkID[pop] = spkID[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
    dspkT[pop] = spkT[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]    
  return simConfig, pdf, actreward, dstartidx, dendidx, dnumc, dspkID, dspkT

def loadInputImages (name=None):
  fn = 'data/'+getsimname(name)+'InputImages.txt'
  print('loading input images from', fn)
  Input_Images = np.loadtxt(fn)
  New_InputImages = []
  NB_Images = int(Input_Images.shape[0]/Input_Images.shape[1])
  for x in range(NB_Images):
    fp = x*Input_Images.shape[1]
    # 20 is sqrt of 400 (20x20 pixels). what is 400? number of ER neurons? or getting rid of counter @ top of screen?
    New_InputImages.append(Input_Images[fp:fp+20,:])
  return np.array(New_InputImages)

def loadMotionFields (name=None): return pickle.load(open('data/'+getsimname(name)+'MotionFields.pkl','rb'))

totalDur = int(dconf['sim']['duration']) # total simulation duration

#
def getrate (dspkT,dspkID, pop, dnumc):
  # get average firing rate for the population, over entire simulation
  nspk = len(dspkT[pop])
  ncell = dnumc[pop]
  rate = 1e3*nspk/(totalDur*ncell)
  return rate

def pravgrates(dspkT,dspkID,dnumc):
  # print average firing rates over simulation duration
  for pop in dspkT.keys(): print(pop,round(getrate(dspkT,dspkID,pop,dnumc),2),'Hz')

#
def drawraster (dspkT,dspkID,tlim=None,msz=2):
  # draw raster (x-axis: time, y-axis: neuron ID)
  csm=cm.ScalarMappable(cmap=cm.prism); csm.set_clim(0,len(dspkT.keys()))
  lclr = []
  for pdx,pop in enumerate(list(dspkT.keys())):
    color = csm.to_rgba(pdx); lclr.append(color)
    plot(dspkT[pop],dspkID[pop],'o',color=color,markersize=msz)
  if tlim is not None:
    xlim(tlim)
  else:
    xlim((0,totalDur))
  xlabel('Time (ms)')
  lclr.reverse(); lpop=list(dspkT.keys()); lpop.reverse()
  lpatch = [mpatches.Patch(color=c,label=s+' '+str(round(getrate(dspkT,dspkID,s,dnumc),2))+' Hz') for c,s in zip(lclr,lpop)]
  ax=gca()
  ax.legend(handles=lpatch,handlelength=1,loc='best')
  

#
def animSynWeights (pdf, outpath='gif/'+dconf['sim']['name']+'weightmap.mp4', framerate=10, figsize=(14,8), cmap='jet'):  
  # animate the synaptic weights along with some stats on behavior
  origfsz = rcParams['font.size']; rcParams['font.size'] = 12; ioff() # save original font size, turn off interactive plotting
  utimes = np.unique(pdf.time)
  minwt = np.min(pdf.weight)
  maxwt = np.max(pdf.weight)
  wtrange = 0.1*(maxwt-minwt)
  print('minwt:',minwt,'maxwt:',maxwt)
  if figsize is not None: fig = plt.figure(figsize=figsize)
  else: fig = plt.figure()
  gs = gridspec.GridSpec(4,8)
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
  pdfsL = pdf[(pdf.postid>=dstartidx['EMDOWN']) & (pdf.postid<=dendidx['EMDOWN'])]
  pdfsR = pdf[(pdf.postid>=dstartidx['EMUP']) & (pdf.postid<=dendidx['EMUP'])]
  Lwts = [np.mean(pdfsL[(pdfsL.time==t)].weight) for t in utimes] #wts of connections onto EMDOWN
  Rwts = [np.mean(pdfsR[(pdfsR.time==t)].weight) for t in utimes] #wts of connections onto EMUP
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
  #plot mean weights of all connections onto EMDOWN and EMUP
  f_ax3.plot(utimes,Lwts,'r-o',markersize=1)
  f_ax3.plot(utimes,Rwts,'b-o',markersize=1)
  f_ax3.set_xlim((0,simConfig['simConfig']['duration']))
  f_ax3.set_ylim((np.min([np.min(Lwts),np.min(Rwts)]),np.max([np.max(Lwts),np.max(Rwts)])))
  f_ax3.set_ylabel('Average weight'); #f_ax2.set_xlabel('Time (ms)')
  f_ax3.legend(('->EMDOWN','->EMUP'),loc='upper left')
  f_ax4.plot(action_times,cumHits,'g-o',markersize=1)
  f_ax4.plot(action_times,cumMissed,'k-o',markersize=1)
  f_ax4.set_xlim((0,np.max(action_times)))
  f_ax4.set_ylim((0,np.max([cumHits[-1],cumMissed[-1]])))
  f_ax4.legend(('Hit Ball','Miss Ball'),loc='upper left')
  lsrc = ['EV1', 'EV4', 'EMT','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE']
  ltitle = []
  for src in lsrc:
    for trg in ['EMDOWN', 'EMUP']: ltitle.append(src+'->'+trg)
  dimg = {}; dline = {}; 
  def getwts (tdx, src):
    t = utimes[tdx]
    ltarg = ['EMDOWN', 'EMUP']
    lout = []
    for targ in ltarg:
      cpdf = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[targ]) & (pdf.postid<=dendidx[targ]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
      lwt = list(np.array(cpdf.weight))
      sz = int(ceil(np.sqrt(len(lwt))))
      sz2 = sz*sz
      while len(lwt) < sz2: lwt.append(lwt[-1])
      lwt = np.reshape(np.array(lwt), (sz,sz))
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
      for trg in ['EMDOWN', 'EMUP']:
          davgw[src+'->'+trg] = arr = []        
          for t in utimes:
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
              arr.append(np.mean(pdfs.weight))
      subplot(12,1,gdx)
      plot(utimes,davgw[src+'->EMDOWN'],'r-',linewidth=3);
      plot(utimes,davgw[src+'->EMUP'],'b-',linewidth=3); 
      legend((src+'->EMDOWN',src+'->EMUP'),loc='upper left')
      plot(utimes,davgw[src+'->EMDOWN'],'ro',markersize=10);
      plot(utimes,davgw[src+'->EMUP'],'bo',markersize=10);       
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
      for trg in ['EMDOWN', 'EMUP']:
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
      plot(utimes,np.array(wperPostID[src+'->EMDOWN']),'r-o',linewidth=3,markersize=5)
      #legend((src+'->EMDOWN'),loc='upper left')
      xlim((0,simConfig['simConfig']['duration']))
      ylabel(src+'->EMDOWN weights')
      subplot(3,1,3)
      plot(utimes,np.array(wperPostID[src+'->EMUP']),'b-o',linewidth=3,markersize=5) 
      #legend((src+'->EMUP'),loc='upper left')       
      xlim((0,simConfig['simConfig']['duration']))
      ylabel(src+'->EMUP weights') 
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
      for trg in ['EMDOWN', 'EMUP']:
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
      imshow(np.transpose(np.array(wperPostID[src+'->EMDOWN'])),aspect = 'auto',cmap='hot', interpolation='None')
      b1 = gca().get_xticks()
      gca().set_xticks(b1-1)
      gca().set_xticklabels((100*b1).astype(int))
      colorbar(orientation='horizontal',fraction=0.05)
      #legend((src+'->EMDOWN'),loc='upper left')
      xlim((-1,b1[-1]-1))
      ylabel(src+'->EMDOWN weights')
      subplot(3,1,3)
      imshow(np.transpose(np.array(wperPostID[src+'->EMUP'])),aspect = 'auto',cmap='hot', interpolation='None') 
      b2 = gca().get_xticks()
      gca().set_xticks(b2-1)
      gca().set_xticklabels((100*b2).astype(int))
      colorbar(orientation='horizontal',fraction=0.05)
      #legend((src+'->EMUP'),loc='upper left')       
      xlim((-1,b2[-1]-1))
      ylabel(src+'->EMUP weights') 
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
      for trg in ['EMDOWN', 'EMUP']:
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
      imshow(np.transpose(np.array(preNeuronIDs[src+'->EMDOWN'])),aspect = 'auto',cmap=c1, interpolation='None')
      subplot(position=[0.15,0.1,0.8,0.8])
      imshow(np.transpose(np.array(allweights[src+'->EMDOWN'])),aspect = 'auto',cmap='hot', interpolation='None')
      b1 = gca().get_xticks()
      gca().set_xticks(b1-1)
      gca().set_xticklabels((100*b1).astype(int))
      colorbar(orientation='vertical',fraction=0.01)
      #legend((src+'->EMDOWN'),loc='upper left')
      xlim((-1,b1[-1]-1))
      ylabel(src+'->EMDOWN weights')
      xlabel('Time (ms)')
      subplot(position=[0.98,0.1,0.01,0.8])
      imshow(np.transpose(np.array(postNeuronIDs[src+'->EMDOWN'])),aspect = 'auto',cmap=c1, interpolation='None')
      #subplot(2,1,2)
      figure()
      subplot(position=[0.05,0.1,0.01,0.8])
      imshow(np.transpose(np.array(preNeuronIDs[src+'->EMUP'])),aspect = 'auto',cmap=c1, interpolation='None')
      subplot(position=[0.15,0.1,0.8,0.8])
      imshow(np.transpose(np.array(allweights[src+'->EMUP'])),aspect = 'auto',cmap='hot', interpolation='None') 
      b2 = gca().get_xticks()
      gca().set_xticks(b2-1)
      gca().set_xticklabels((100*b2).astype(int))
      colorbar(orientation='vertical',fraction=0.01)
      #legend((src+'->EMUP'),loc='upper left')       
      xlim((-1,b2[-1]-1))
      ylabel(src+'->EMUP weights') 
      xlabel('Time (ms)')
      subplot(position=[0.98,0.1,0.01,0.8])
      imshow(np.transpose(np.array(postNeuronIDs[src+'->EMUP'])),aspect = 'auto',cmap=c1, interpolation='None')

def plotSynWeightsPostNeuronID(pdf,postNeuronID):
  utimes = np.unique(pdf.time)
  #for a postID, find a neuron in ML and a neuron in MUP
  pdfs_MDOWN = pdf[(pdf.time==utimes[0]) & (pdf.postid>=dstartidx['EMDOWN']) & (pdf.postid<=dendidx['EMDOWN'])]
  uIDs_MDOWN = np.unique(pdfs_MDOWN.postid)
  pdfs_MUP = pdf[(pdf.time==utimes[0]) & (pdf.postid>=dstartidx['EMUP']) & (pdf.postid<=dendidx['EMUP'])]
  uIDs_MUP = np.unique(pdfs_MUP.postid)
  targetMDOWN_postID = min(uIDs_MDOWN)-1+postNeuronID
  targetMUP_postID = min(uIDs_MUP)-1+postNeuronID

  NBpreN_MDOWN = len(np.unique(pdfs_MDOWN.preid))
  NBpreN_MUP = len(np.unique(pdfs_MUP.preid)) 
  #for every postsynaptic neuron, find total weight of synaptic inputs per area (i.e. synaptic inputs from EV1, EV4 and EIT and treated separately for each cell——if there are 200 unique cells, will get 600 weights as 200 from each originating layer)
  MDOWNweights = {}
  MUPweights = {}
  MDOWNpreNeuronIDs = {}
  MUPpreNeuronIDs = {}
  
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
      MDOWNweights[src] = arrL = []
      MUPweights[src] = arrR = []
      MDOWNpreNeuronIDs[src] = arrL2 = []
      MUPpreNeuronIDs[src] = arrR2 = []
      tstep = 0
      for t in utimes:
          arrL.append([])
          arrR.append([])
          arrL2.append([])
          arrR2.append([])
          pdfsL = pdf[(pdf.time==t) & (pdf.postid==targetMDOWN_postID) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
          pdfsR = pdf[(pdf.time==t) & (pdf.postid==targetMUP_postID) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
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
      plot(utimes,np.array(MDOWNweights[src]),'r-o',linewidth=3,markersize=5)
      plot(utimes,np.array(MUPweights[src]),'b-o',linewidth=3,markersize=5)
      legend((src+'->EMDOWN',src+'->EMUP'),loc='upper left')
      xlim((0,simConfig['simConfig']['duration']))
      pdx += 1        

if __name__ == '__main__':
  stepNB = -1
  if len(sys.argv) > 1:
    try:
      stepNB = int(sys.argv[1]) #which file(stepNB) want to plot
    except:
      pass
  print(stepNB)
  simConfig, pdf, actreward, dstartidx, dendidx, dnumc, dspkID, dspkT = loadsimdat()
  print('loaded simulation data')
  #davgw = plotavgweights(pdf)
  #animSynWeights(pdf,'gif/'+dconf['sim']['name']+'weightmap.mp4', framerate=10) #plot/save images as movie
  #wperPostID = plotavgweightsPerPostSynNeuron1(pdf)
  #plotavgweightsPerPostSynNeuron2(pdf)
  #plotIndividualSynWeights(pdf)
  #plotSynWeightsPostNeuronID(pdf,5)
  #plotSynWeightsPostNeuronID(pdf,15)
  #plotSynWeightsPostNeuronID(pdf,25)
  #plotSynWeightsPostNeuronID(pdf,35)
  #plotSynWeightsPostNeuronID(pdf,45)
