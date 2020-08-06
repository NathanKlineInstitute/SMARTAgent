import numpy as np
from pylab import *
import pickle
import pandas as pd
import conf
from conf import dconf
import os
import sys
import anim
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict
from imgutils import getoptflow, getoptflowframes
from connUtils import gid2pos
from utils import getdatestr
from scipy.stats import pearsonr

rcParams['agg.path.chunksize'] = 100000000000 # for plots of long activity 
ion()

rcParams['font.size'] = 12
tl=tight_layout
stepNB = -1
totalDur = int(dconf['sim']['duration']) # total simulation duration
allpossible_pops = ['ER','IR','EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE','IV1','EV4','IV4','EMT','IMT','EMDOWN','EMUP','EMSTAY','IM']

def pdf2weightsdict (pdf):
  # convert the pandas dataframe with synaptic weights into a dictionary
  D = {}
  for idx in pdf.index:
    t, preid, postid, weight = pdf.loc[idx]
    preid=int(preid)
    postid=int(postid)
    if preid not in D: D[preid] = {}
    if postid not in D[preid]: D[preid][postid] = []
    D[preid][postid].append([t,weight])
  return D

def readweightsfile2pdf (fn):
  # read the synaptic plasticity weights saved as a dictionary into a pandas dataframe  
  D = pickle.load(open(fn,'rb'))
  A = []
  for preID in D.keys():
    for poID in D[preID].keys():
      for row in D[preID][poID]:
        A.append([row[0], preID, poID, row[1]])
  return pd.DataFrame(A,columns=['time','preid','postid','weight'])

#
def readinweights (name,final=False):
  # read the synaptic plasticity weights associated with sim name into a pandas dataframe
  if final:
    fn = 'data/'+name+'synWeights_final.pkl'
  else:
    fn = 'data/'+name+'synWeights.pkl'
  return readweightsfile2pdf(fn)

def savefinalweights (pdf, simstr):
  # save final weights to a (small) file
  pdfs = pdf[pdf.time==np.amax(pdf.time)]
  D = pdf2weightsdict(pdfs)
  pickle.dump(D, open('data/'+simstr+'synWeights_final.pkl','wb'))  

def getsimname (name=None):
  if name is None:
    if stepNB is -1: name = dconf['sim']['name']
    elif stepNB > -1: name = dconf['sim']['name'] + '_step_' + str(stepNB) + '_'
  return name

def generateActivityMap(t1, t2, spkT, spkID, numc, startidx):
  sN = int(np.sqrt(numc))
  Nact = np.zeros(shape=(len(t1),sN,sN)) # Nact is 3D array of number of spikes, indexed by: time, y, x
  for i in range(sN):
    for j in range(sN):
      cNeuronID = j+(i*sN) + startidx
      cNeuron_spkT = spkT[spkID==cNeuronID]
      for t in range(len(t1)):
        cbinSpikes = cNeuron_spkT[(cNeuron_spkT>t1[t]) & (cNeuron_spkT<=t2[t])]
        Nact[t][i][j] = len(cbinSpikes)
  return Nact

def getdActMap (totalDur, tstepPerAction, dspkT, dspkID, dnumc, dstartidx,lpop = allpossible_pops):
  t1 = range(0,totalDur,tstepPerAction)
  t2 = range(tstepPerAction,totalDur+tstepPerAction,tstepPerAction)
  dact = {}
  for pop in lpop:
    if pop in dnumc and dnumc[pop] > 0:
      dact[pop] = generateActivityMap(t1, t2, dspkT[pop], dspkID[pop], dnumc[pop], dstartidx[pop])
  return dact
  
def loadsimdat (name=None,getactmap=True,lpop = allpossible_pops): # load simulation data
  global totalDur, tstepPerAction
  name = getsimname(name)
  print('loading data from', name)
  conf.dconf = conf.readconf('backupcfg/'+name+'sim.json')
  simConfig = pickle.load(open('data/'+name+'simConfig.pkl','rb'))
  dstartidx,dendidx={},{} # starting,ending indices for each population
  for p in simConfig['net']['pops'].keys():
    if simConfig['net']['pops'][p]['tags']['numCells'] > 0:
      dstartidx[p] = simConfig['net']['pops'][p]['cellGids'][0]
      dendidx[p] = simConfig['net']['pops'][p]['cellGids'][-1]
  pdf=None
  try:
    pdf = readinweights(name) # if RL was off, no weights saved
  except:
    try:
      pdf = readinweights(name,final=True)
    except:
      pass
  actreward = pd.DataFrame(np.loadtxt('data/'+name+'ActionsRewards.txt'),columns=['time','action','reward','proposed','hit'])
  dnumc = {}
  for p in simConfig['net']['pops'].keys():
    if p in dstartidx:
      dnumc[p] = dendidx[p]-dstartidx[p]+1
    else:
      dnumc[p] = 0
  spkID= np.array(simConfig['simData']['spkid'])
  spkT = np.array(simConfig['simData']['spkt'])
  dspkID,dspkT = {},{}
  for pop in simConfig['net']['pops'].keys():
    if dnumc[pop] > 0:
      dspkID[pop] = spkID[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
      dspkT[pop] = spkT[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
  InputImages=ldflow=None
  InputImages = loadInputImages(dconf['sim']['name'])
  ldflow = loadMotionFields(dconf['sim']['name'])
  totalDur = int(dconf['sim']['duration'])
  tstepPerAction = dconf['sim']['tstepPerAction'] # time step per action (in ms)  
  dact = None
  #lpop = ['ER', 'EV1', 'EV4', 'EMT', 'IR', 'IV1', 'IV4', 'IMT',\
  #        'EV1DW','EV1DNW', 'EV1DN', 'EV1DNE','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE',\
  #        'EMDOWN','EMUP']  
  if getactmap: dact = getdActMap(totalDur, tstepPerAction, dspkT, dspkID, dnumc, dstartidx, lpop)
  return simConfig, pdf, actreward, dstartidx, dendidx, dnumc, dspkID, dspkT, InputImages, ldflow, dact

#
def animActivityMaps (outpath='gif/'+dconf['sim']['name']+'actmap.mp4', framerate=10, figsize=(18,10), dobjpos=None,\
                      lpop=allpossible_pops, nframe=None):
  # plot activity in different layers as a function of input images  
  ioff()
  possible_pops = ['ER','EV1','EV4','EMT','IR','IV1','IV4','IMT','EV1DW','EV1DNW','EV1DN'\
                   ,'EV1DNE','EV1DE','EV1DSW','EV1DS', 'EV1DSE','EMDOWN','EMUP','EMSTAY','IM']
  possible_titles = ['Excit R', 'Excit V1', 'Excit V4', 'Excit MT', 'Inhib R', 'Inhib V1', 'Inhib V4', 'Inhib MT',\
                     'W','NW','N','NE','E','SW','S','SE','Excit M DOWN', 'Excit M UP', 'Excit M STAY', 'Inhib M']
  dtitle = {p:t for p,t in zip(possible_pops,possible_titles)}
  ltitle = ['Input Images']
  lact = [InputImages]; lvmax = [255];
  dmaxSpk = OrderedDict({pop:np.max(dact[pop]) for pop in dact.keys()})
  max_spks = np.max([dmaxSpk[p] for p in dact.keys()])
  for pop in lpop:
    ltitle.append(dtitle[pop])
    lact.append(dact[pop])
    lvmax.append(max_spks)
  if figsize is not None: fig, axs = plt.subplots(4, 5, figsize=figsize);
  else: fig, axs = plt.subplots(4, 5);
  lax = axs.ravel()
  cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2])  
  ddir = OrderedDict({'EV1DW':'W','EV1DNW':'NW', 'EV1DN':'N','EV1DNE':'NE','EV1DE':'E','EV1DSW':'SW','EV1DS':'S','EV1DSE':'SE'})
  lfnimage = []
  ddat = {}
  fig.suptitle('Time = ' + str(0*tstepPerAction) + ' ms')
  idx = 0
  objfctr = 1.0
  if 'UseFull' in dconf['DirectionDetectionAlgo']:
    if dconf['DirectionDetectionAlgo']['UseFull']: objfctr=1/8.
  flowdx = 8 # 5
  for ldx,ax in enumerate(lax):
    if idx > len(dact.keys()):
      ax.axis('off')
      continue
    if ldx==0:
      offidx=-1
    elif ldx==flowdx:
      offidx=1
    else:
      offidx=0
    if ldx==flowdx:
      X, Y = np.meshgrid(np.arange(0, InputImages[0].shape[1], 1), np.arange(0,InputImages[0].shape[0],1))
      ddat[ldx] = ax.quiver(X,Y,ldflow[0]['thflow'][:,:,0],-ldflow[0]['thflow'][:,:,1], pivot='mid', units='inches',width=0.022,scale=1/0.15)
      ax.set_xlim((0,InputImages[0].shape[1])); ax.set_ylim((0,InputImages[0].shape[0]))
      ax.invert_yaxis()              
      continue
    else:
      pcm = ax.imshow(lact[idx][offidx,:,:],origin='upper',cmap='gray',vmin=0,vmax=lvmax[idx])
      ddat[ldx] = pcm
      if ldx==0 and dobjpos is not None:
        lobjx,lobjy = [objfctr*dobjpos[k][0,0] for k in dobjpos.keys()], [objfctr*dobjpos[k][0,1] for k in dobjpos.keys()]
        ddat['objpos'], = ax.plot(lobjx,lobjy,'ro')      
    ax.set_ylabel(ltitle[idx])
    if ldx==2: plt.colorbar(pcm, cax = cbaxes)  
    idx += 1
  def updatefig (t):
    fig.suptitle('Time = ' + str(t*tstepPerAction) + ' ms')    
    if t<1: return fig # already rendered t=0 above; skip last for optical flow
    print('frame t = ', str(t*tstepPerAction))
    idx = 0
    for ldx,ax in enumerate(lax):
      if idx > len(dact.keys()): continue
      if ldx==0 or ldx==flowdx:
        offidx=-1
      else:
        offidx=0
      if ldx == flowdx:
        ddat[ldx].set_UVC(ldflow[t+offidx]['thflow'][:,:,0],-ldflow[t]['thflow'][:,:,1])        
      else:
        ddat[ldx].set_data(lact[idx][t+offidx,:,:])
        if ldx==0 and dobjpos is not None:
          lobjx,lobjy = [objfctr*dobjpos[k][t,0] for k in dobjpos.keys()], [objfctr*dobjpos[k][t,1] for k in dobjpos.keys()]
          ddat['objpos'].set_data(lobjx,lobjy)        
        idx += 1
    return fig
  t1 = range(0,totalDur,tstepPerAction)
  if nframe is None: nframe = len(t1) - 1
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=nframe)
  writer = anim.getwriter(outpath, framerate=framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  ion()
  return fig, axs, plt

#
def animInput (InputImages, outpath, framerate=10, figsize=None, showflow=False, ldflow=None, dobjpos=None):
  # animate the input images; showflow specifies whether to calculate/animate optical flow
  ioff()
  # plot input images and optionally optical flow
  ncol = 1
  if showflow: ncol+=1
  if figsize is not None: fig = figure(figsize=figsize)
  else: fig = figure()
  lax = [subplot(1,ncol,i+1) for i in range(ncol)]
  ltitle = ['Input Images']
  lact = [InputImages]; lvmax = [255]; xl = [(-.5,19.5)]; yl = [(19.5,-0.5)]
  ddat = {}
  fig.suptitle('Time = ' + str(0*tstepPerAction) + ' ms')
  idx = 0
  lflow = []
  if showflow and ldflow is None: ldflow = getoptflowframes(InputImages)
  objfctr = 1.0
  if 'UseFull' in dconf['DirectionDetectionAlgo']:
    if dconf['DirectionDetectionAlgo']['UseFull']: objfctr=1/8.
  for ldx,ax in enumerate(lax):
    if ldx==0:
      pcm = ax.imshow( lact[idx][0,:,:], origin='upper', cmap='gray', vmin=0, vmax=lvmax[idx])
      ddat[ldx] = pcm
      ax.set_ylabel(ltitle[idx])
      if dobjpos is not None:
        lobjx,lobjy = [objfctr*dobjpos[k][0,0] for k in dobjpos.keys()], [objfctr*dobjpos[k][0,1] for k in dobjpos.keys()]
        ddat['objpos'], = ax.plot(lobjx,lobjy,'ro')
    else:
      X, Y = np.meshgrid(np.arange(0, InputImages[0].shape[1], 1), np.arange(0,InputImages[0].shape[0],1))
      ddat[ldx] = ax.quiver(X,Y,ldflow[0]['thflow'][:,:,0],-ldflow[0]['thflow'][:,:,1], pivot='mid', units='inches',width=0.01,scale=1/0.3)
      ax.set_xlim((0,InputImages[0].shape[1])); ax.set_ylim((0,InputImages[0].shape[0]))
      ax.invert_yaxis()
    idx += 1
  def updatefig (t):
    fig.suptitle('Time = ' + str(t*tstepPerAction) + ' ms')
    if t < 1: return fig # already rendered t=0 above
    print('frame t = ', str(t*tstepPerAction))    
    for ldx,ax in enumerate(lax):
      if ldx == 0:
        ddat[ldx].set_data(lact[0][t,:,:])
        if dobjpos is not None:
          lobjx,lobjy = [objfctr*dobjpos[k][t,0] for k in dobjpos.keys()], [objfctr*dobjpos[k][t,1] for k in dobjpos.keys()]
          ddat['objpos'].set_data(lobjx,lobjy)
      else:
        ddat[ldx].set_UVC(ldflow[t-1]['thflow'][:,:,0],-ldflow[t]['thflow'][:,:,1])        
    return fig
  t1 = range(0,totalDur,tstepPerAction)
  nframe = len(t1)
  if showflow: nframe-=1
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=nframe)
  writer = anim.getwriter(outpath, framerate=framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  ion()
  return fig

#
def getmaxdir (dact, ddir):
  ddir = OrderedDict({'EV1DW':'W','EV1DNW':'NW', 'EV1DN':'N','EV1DNE':'NE','EV1DE':'E','EV1DSW':'SW','EV1DS':'S','EV1DSE':'SE'})
  maxdirX = np.zeros(dact['EV1DW'].shape)
  maxdirY = np.zeros(dact['EV1DW'].shape)
  dAngDir = OrderedDict({'EV1DE': [1,0],'EV1DNE': [np.sqrt(2),np.sqrt(2)], # receptive field peak angles for the direction selective populations
                            'EV1DN': [0,1],'EV1DNW': [-np.sqrt(2),np.sqrt(2)],
                            'EV1DW': [-1,0],'EV1DSW': [-np.sqrt(2),-np.sqrt(2)],
                            'EV1DS': [0,-1],'EV1DSE': [np.sqrt(2),-np.sqrt(2)],
                            'NOMOVE':[0,0]})
  for k in dAngDir.keys():
    dAngDir[k][0] *= .4
    dAngDir[k][1] *= -.4
  for tdx in range(maxdirX.shape[0]):
    for y in range(maxdirX.shape[1]):
      for x in range(maxdirX.shape[2]):
        maxval = 0
        maxdir = 'NOMOVE'
        for pop in ddir.keys():
          if dact[pop][tdx,y,x] > maxval:
            maxval = dact[pop][tdx,y,x]
            maxdir = pop
        maxdirX[tdx,y,x] = dAngDir[maxdir][0]
        maxdirY[tdx,y,x] = dAngDir[maxdir][1]
  return maxdirX,maxdirY

#
def animDetectedMotionMaps (outpath, framerate=10, figsize=(7,3)):
  ioff()
  # plot activity in different layers as a function of input images
  ddir = OrderedDict({'EV1DW':'W','EV1DNW':'NW', 'EV1DN':'N','EV1DNE':'NE','EV1DE':'E','EV1DSW':'SW','EV1DS':'S','EV1DSE':'SE'})
  if figsize is not None: fig, axs = plt.subplots(1, 3, figsize=figsize);
  else: fig, axs = plt.subplots(1, 3);
  lax = axs.ravel()
  ltitle = ['Input Images', 'Motion', 'Detected Motion']
  lact = [InputImages]; lvmax = [255];
  lfnimage = []
  lpop = ['ER', 'EV1', 'EV4', 'EMT', 'IR', 'IV1', 'IV4', 'IMT',\
          'EV1DW','EV1DNW', 'EV1DN', 'EV1DNE','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE',\
          'EMDOWN','EMUP','EMSTAY']  
  dmaxSpk = OrderedDict({pop:np.max(dact[pop]) for pop in lpop})
  max_spks = np.max([dmaxSpk[p] for p in lpop])
  for pop in lpop:
    lact.append(dact[pop])
    lvmax.append(max_spks)
  ddat = {}
  fig.suptitle('Time = ' + str(0*tstepPerAction) + ' ms')
  maxdirX,maxdirY = getmaxdir(dact,ddir)
  for ldx,ax in enumerate(lax):
    if ldx == 0:
      offidx = -1
      pcm = ax.imshow(lact[0][offidx,:,:],origin='upper',cmap='gray',vmin=0,vmax=lvmax[0])
      ddat[ldx] = pcm            
    elif ldx == 1:
      X, Y = np.meshgrid(np.arange(0, InputImages[0].shape[1], 1), np.arange(0,InputImages[0].shape[0],1))
      ddat[ldx] = ax.quiver(X,Y,ldflow[0]['thflow'][:,:,0],-ldflow[0]['thflow'][:,:,1], pivot='mid', units='inches',width=0.022,scale=1/0.15)
      ax.set_xlim((0,InputImages[0].shape[1])); ax.set_ylim((0,InputImages[0].shape[0]))
      ax.invert_yaxis()                    
    elif ldx == 2:
      X, Y = np.meshgrid(np.arange(0, InputImages[0].shape[1], 1), np.arange(0,InputImages[0].shape[0],1))
      ddat[ldx] = ax.quiver(X,Y,maxdirX[0,:,:],maxdirY[0,:,:], pivot='mid', units='inches',width=0.022,scale=1/0.15)
      ax.set_xlim((0,InputImages[0].shape[1])); ax.set_ylim((0,InputImages[0].shape[0]))
      ax.invert_yaxis()
    ax.set_ylabel(ltitle[ldx])
  def updatefig (t):
    fig.suptitle('Time = ' + str(t*tstepPerAction) + ' ms')    
    if t<1: return fig # already rendered t=0 above; skip last for optical flow
    print('frame t = ', str(t*tstepPerAction))
    for ldx,ax in enumerate(lax):
      if ldx==0 or ldx==5:
        offidx=-1
      else:
        offidx=0
      if ldx == 0:
        ddat[ldx].set_data(lact[0][t+offidx,:,:])
      elif ldx == 1:
        ddat[ldx].set_UVC(ldflow[t+offidx]['thflow'][:,:,0],-ldflow[t]['thflow'][:,:,1])        
      else:
        ddat[ldx].set_UVC(maxdirX[t+offidx,:,:],maxdirY[t+offidx,:,:])
    return fig
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=len(t1)-1)
  writer = anim.getwriter(outpath, framerate=framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  ion()
  return fig, axs, plt

def loadInputImages (name=None):
  fn = 'data/'+getsimname(name)+'InputImages.txt'
  print('loading input images from', fn)
  Input_Images = np.loadtxt(fn)
  New_InputImages = []
  NB_Images = int(Input_Images.shape[0]/Input_Images.shape[1])
  for x in range(NB_Images):
    fp = x*Input_Images.shape[1]
    # 20 is sqrt of 400 (20x20 pixels). what is 400? number of ER neurons? or getting rid of counter @ top of screen?
    New_InputImages.append(Input_Images[fp:fp+Input_Images.shape[1],:])
  return np.array(New_InputImages)

def loadMotionFields (name=None): return pickle.load(open('data/'+getsimname(name)+'MotionFields.pkl','rb'))

def loadObjPos (name=None): return pickle.load(open('data/'+getsimname(name)+'objpos.pkl','rb'))

def getspikehist (spkT, dnumc, binsz, tmax):
  tt = np.arange(0,tmax,binsz)
  nspk = [len(spkT[(spkT>=tstart) & (spkT<tstart+binsz)]) for tstart in tt]
  return tt,nspk

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
def drawcellVm (simConfig, ldrawpop=None,tlim=None):
  csm=cm.ScalarMappable(cmap=cm.prism); csm.set_clim(0,len(dspkT.keys()))
  if tlim is not None:
    dt = simConfig['simData']['t'][1]-simConfig['simData']['t'][0]    
    sidx,eidx = int(0.5+tlim[0]/dt),int(0.5+tlim[1]/dt)
  dclr = OrderedDict(); lpop = []
  for kdx,k in enumerate(list(simConfig['simData']['V_soma'].keys())):  
    color = csm.to_rgba(kdx); 
    cty = simConfig['net']['cells'][int(k.split('_')[1])]['tags']['cellType']
    if ldrawpop is not None and cty not in ldrawpop: continue
    dclr[kdx]=color
    lpop.append(simConfig['net']['cells'][int(k.split('_')[1])]['tags']['cellType'])
  if ldrawpop is None: ldrawpop = lpop    
  for kdx,k in enumerate(list(simConfig['simData']['V_soma'].keys())):
    cty = simConfig['net']['cells'][int(k.split('_')[1])]['tags']['cellType']
    if ldrawpop is not None and cty not in ldrawpop: continue
    if tlim is not None:
      plot(simConfig['simData']['t'][sidx:eidx],simConfig['simData']['V_soma'][k][sidx:eidx],color=dclr[kdx])
    else:
      plot(simConfig['simData']['t'],simConfig['simData']['V_soma'][k],color=dclr[kdx])      
  lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(dclr.values(),ldrawpop)]
  ax=gca()
  ax.legend(handles=lpatch,handlelength=1,loc='best')
  if tlim is not None: ax.set_xlim(tlim)
  
#  
def plotFollowBall (actreward, ax=None,cumulative=True,msz=3,binsz=1e3,color='r'):
  # plot probability of model racket following target(predicted ball y intercept) vs time
  # when cumulative == True, plots cumulative probability; otherwise bins probabilities over binsz interval
  # not a good way to plot probabilities over time when uneven sampling - could resample to uniform intervals ...
  # for now cumulative == False is not plotted at all ... 
  global tstepPerAction
  if ax is None: ax = gca()
  ax.plot([0,np.amax(actreward.time)],[0.5,0.5],'--',color='gray')    
  allproposed = actreward[(actreward.proposed!=-1)] # only care about cases when can suggest a proposed action
  rewardingActions = np.where(allproposed.proposed-allproposed.action==0,1,0)
  if cumulative:
    rewardingActions = np.cumsum(rewardingActions) # cumulative of rewarding action
    cumActs = np.array(range(1,len(allproposed)+1))
    aout = np.divide(rewardingActions,cumActs)
    ax.plot(allproposed.time,aout,color+'.',markersize=msz)
  else:
    nbin = int(binsz / (np.array(actreward.time)[1]-np.array(actreward.time)[0]))
    aout = avgfollow = [mean(rewardingActions[sidx:sidx+nbin]) for sidx in arange(0,len(rewardingActions),nbin)]
    ax.plot(np.linspace(0,np.amax(actreward.time),len(avgfollow)), avgfollow, color,linewidth=msz)
  ax.set_xlim((0,np.amax(actreward.time)))
  ax.set_ylim((0,1))
  ax.set_xlabel('Time (ms)'); ax.set_ylabel('p(Follow Target)')
  return aout


def getCumScore (actreward):
  # get cumulative score - assumes score has max reward
  ScoreLoss = np.array(actreward.reward)
  allScore = np.where(ScoreLoss==dconf['rewardcodes']['scorePoint'],1,0) 
  return np.cumsum(allScore) #cumulative score evolving with time.  

#  
def plotHitMiss (actreward,ax=None,msz=3,asratio=False,asbin=False,binsz=10e3,lclr=['r','g','b']):
  if ax is None: ax = gca()
  action_times = np.array(actreward.time)
  Hit_Missed = np.array(actreward.hit)
  allHit = np.where(Hit_Missed==1,1,0) 
  allMissed = np.where(Hit_Missed==-1,1,0)
  cumHits = np.cumsum(allHit) #cumulative hits evolving with time.
  cumMissed = np.cumsum(allMissed) #if a reward is -1, replace it with 1 else replace it with 0.
  if asbin:
    nbin = int(binsz / (np.array(actreward.time)[1]-np.array(actreward.time)[0]))
    avgHit = np.array([sum(allHit[sidx:sidx+nbin]) for sidx in arange(0,len(allHit),nbin)])
    avgMiss = np.array([sum(allMissed[sidx:sidx+nbin]) for sidx in arange(0,len(allMissed),nbin)])
    score = avgHit / (avgHit + avgMiss)
    ax.plot(np.linspace(0,np.amax(actreward.time),len(score)), score, color=lclr[0],linewidth=msz)
    ax.set_ylabel('Hit/(Hit+Miss) ('+str(round(score[-1],2))+')')    
    return score
  elif asratio:
    ax.plot(action_times,cumHits/cumMissed,lclr[0]+'-o',markersize=msz)
    ax.set_xlim((0,np.max(action_times)))
    ax.set_ylabel('Hit/Miss ('+str(round(cumHits[-1]/cumMissed[-1],2))+')')
    return cumHits[-1]/cumMissed[-1]
  else:
    ax.plot(action_times,cumHits,lclr[0]+'-o',markersize=msz)
    ax.plot(action_times,cumMissed,lclr[1]+'-o',markersize=msz)
    ax.set_xlim((0,np.max(action_times)))
    ax.set_ylim((0,np.max([cumHits[-1],cumMissed[-1]])))    
    ax.set_ylabel('Hit Ball ('+str(cumHits[-1])+')','Miss Ball ('+str(cumMissed[-1])+')')
    return cumHits[-1],cumMissed[-1]

#  
def plotScoreMiss (actreward,ax=None,msz=3,asratio=False,clr='r'):
  if ax is None: ax = gca()
  action_times = np.array(actreward.time)
  Hit_Missed = np.array(actreward.hit)
  allMissed = np.where(Hit_Missed==-1,1,0)
  cumMissed = np.cumsum(allMissed) #if a reward is -1, replace it with 1 else replace it with 0.
  cumScore = getCumScore(actreward)
  if asratio:
    ax.plot(action_times,cumScore/cumMissed,clr+'-o',markersize=msz)
    ax.set_xlim((0,np.max(action_times)))
    ax.set_ylabel('Score/Miss ('+str(round(cumScore[-1]/cumMissed[-1],2))+')')
    return cumScore[-1]/cumMissed[-1]
  else:
    ax.plot(action_times,cumScore,clr+'-o',markersize=msz)
    ax.set_xlim((0,np.max(action_times)))
    ax.set_ylim((0,cumMissed[-1]))
    ax.set_ylabel('Score ('+str(cumScore[-1])+')')
    return cumScore[-1]

#  
def plotScoreLoss (actreward,ax=None,msz=3):
  # plot cumulative score points and lose points; assumes score/lose point is max/min reward
  if ax is None: ax = gca()
  action_times = np.array(actreward.time)
  ScoreLoss = np.array(actreward.reward)
  allScore = np.where(ScoreLoss==np.amax(ScoreLoss),1,0) 
  allLoss = np.where(ScoreLoss==np.amin(ScoreLoss),1,0)
  cumScore = np.cumsum(allScore) #cumulative hits evolving with time.
  cumLoss = np.cumsum(allLoss) #if a reward is -1, replace it with 1 else replace it with 0.
  ax.plot(action_times,cumScore,'r-o',markersize=msz)
  ax.plot(action_times,cumLoss,'b-o',markersize=msz)
  ax.set_xlim((0,np.max(action_times)))
  ax.set_ylim((0,np.max([cumScore[-1],cumLoss[-1]])))
  ax.legend(('Score Point ('+str(cumScore[-1])+')','Lose Point ('+str(cumLoss[-1])+')'),loc='best')
  return cumScore[-1],cumLoss[-1]

def plotPerf (actreward,yl=(0,1)):
  # plot performance
  plotFollowBall(actreward,ax=subplot(1,1,1),cumulative=True,color='b');  
  plotHitMiss(actreward,ax=subplot(1,1,1),lclr=['g'],asratio=True); 
  plotScoreMiss(actreward,ax=subplot(1,1,1),clr='r',asratio=True);
  ylim(yl)
  ylabel('Performance')
  lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(['b','g','r'],['Follow','Hit/Miss','Score/Miss'])]
  ax=gca()
  ax.legend(handles=lpatch,handlelength=1)
  return ax

def plotComparePerf (lpda, lclr, yl=(0,.55), lleg=None):
  # plot comparison of performance of list of action rewards dataframes in lpda
  # lclr is color to plot
  # lleg is optional legend
  for pda,clr in zip(lpda,lclr):
    plotFollowBall(pda,ax=subplot(1,3,1),cumulative=True,color=clr); ylim(yl)
    plotHitMiss(pda,ax=subplot(1,3,2),lclr=[clr],asratio=True); ylim(yl)
    plotScoreMiss(pda,ax=subplot(1,3,3),clr=clr,asratio=True); ylim(yl)
  if lleg is not None:
    lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(lclr,lleg)]
    ax=gca()
    ax.legend(handles=lpatch,handlelength=1)
  

#
def plotRewards (actreward,ax=None,msz=3,xl=None):
  if ax is None: ax = gca()  
  ax.plot(actreward.time,actreward.reward,'ko-',markersize=msz)
  if xl is not None: ax.set_xlim(xl)
  ax.set_ylim((np.min(actreward.reward),np.max(actreward.reward)))
  ax.set_ylabel('Rewards'); #f_ax1.set_xlabel('Time (ms)')

def getactsel (dhist, actreward):
  # get action selected based on firing rates in dhist (no check for consistency with sim.py)
  actsel = []
  for i in range(len(dhist['EMDOWN'][1])):
    dspk, uspk, sspk = dhist['EMDOWN'][1][i], dhist['EMUP'][1][i], dhist['EMSTAY'][1][i]
    if dspk > uspk and dspk > sspk:
      actsel.append(dconf['moves']['DOWN'])
    elif uspk > dspk and uspk > sspk:
      actsel.append(dconf['moves']['UP'])
    elif sspk > uspk and sspk > dspk:
      actsel.append(dconf['moves']['NOMOVE'])
    else: # dspk == uspk:
      actsel.append(dconf['moves']['NOMOVE'])
  return actsel  
  
def getconcatactionreward (lfn):
  # concatenate the actionreward data frames together so can look at cumulative rewards,actions,etc.
  # lfn is a list of actionrewards filenames from the simulation
  pda = None
  for fn in lfn:
    if not fn.endswith('ActionsRewards.txt'): fn = 'data/'+fn+'ActionsRewards.txt'
    acl = pd.DataFrame(np.loadtxt(fn),columns=['time','action','reward','proposed','hit'])
    if pda is None:
      pda = acl
    else:
      acl.time += np.amax(pda.time)
      pda = pda.append(acl)
  return pda

def getindivactionreward (lfn):
  # get the individual actionreward data frames separately so can compare cumulative rewards,actions,etc.
  # lfn is a list of actionrewards filenames from the simulation or list of simulation names
  if lfn[0].endswith('ActionsRewards.txt'): 
    return [pd.DataFrame(np.loadtxt(fn),columns=['time','action','reward','proposed','hit']) for fn in lfn]
  else:
    return [pd.DataFrame(np.loadtxt('data/'+fn+'ActionsRewards.txt'),columns=['time','action','reward','proposed','hit']) for fn in lfn]    

def plotMeanNeuronWeight (pdf,postid,clr='k',ax=None,msz=1,xl=None):
  if ax is None: ax = gca()
  utimes = np.unique(pdf.time)
  mnw,mxw=1e9,-1e9
  pdfs = pdf[(pdf.postid==postid) & (pdf.postid==postid)]
  wts = [np.mean(pdfs[(pdfs.time==t)].weight) for t in utimes] #wts of connections onto pop
  ax.plot(utimes,wts,clr+'-o',markersize=msz)
  mnw=min(mnw, min(wts))
  mxw=max(mxw, max(wts))
  if xl is not None: ax.set_xlim(xl)
  ax.set_ylim((mnw,mxw))
  ax.set_ylabel('Average weight'); 
  return wts    
  
def plotMeanWeights (pdf,ax=None,msz=1,xl=None,lpop=['EMDOWN','EMUP','EMSTAY'],lclr=['r','b','g'],plotindiv=True):
  #plot mean weights of all plastic synaptic weights onto lpop
  if ax is None: ax = gca()
  utimes = np.unique(pdf.time)
  popwts = {}
  mnw,mxw=1e9,-1e9
  for pop,clr in zip(lpop,lclr):
    #print(pop,clr)
    if pop in dstartidx:
      if plotindiv:
        for idx in range(dstartidx[pop],dendidx[pop]+1,1): # first plot average weight onto each individual neuron
          lwt = plotMeanNeuronWeight(pdf,idx,clr=clr,msz=1)
          mnw=min(mnw, min(lwt))
          mxw=max(mxw, max(lwt))    
      pdfs = pdf[(pdf.postid>=dstartidx[pop]) & (pdf.postid<=dendidx[pop])]
      popwts[pop] = [np.mean(pdfs[(pdfs.time==t)].weight) for t in utimes] #wts of connections onto pop
      ax.plot(utimes,popwts[pop],clr+'-o',markersize=msz)
      mnw=min(mnw, np.amin(popwts[pop]))
      mxw=max(mxw, np.amax(popwts[pop]))            
  if xl is not None: ax.set_xlim(xl)
  ax.set_ylim((mnw,mxw))
  ax.set_ylabel('Average weight'); 
  ax.legend(handles=[mpatches.Patch(color=c,label=s) for c,s in zip(lclr,lpop)],handlelength=1,loc='best')
  return popwts

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
  gs = gridspec.GridSpec(5,9)
  f_ax = []
  ax_count = 0
  for rows in range(4):
    for cols in range(9): 
      if ax_count<33: 
        f_ax.append(fig.add_subplot(gs[rows,cols]))
      ax_count += 1
  cbaxes = fig.add_axes([0.92, 0.4, 0.01, 0.2])
  f_ax1 = fig.add_subplot(gs[3,6:8])
  f_ax2 = fig.add_subplot(gs[4,0:2])
  f_ax3 = fig.add_subplot(gs[4,3:5])
  f_ax4 = fig.add_subplot(gs[4,6:8])
  plotFollowBall(actreward,f_ax1)
  plotRewards(actreward,f_ax2,xl=(0,simConfig['simConfig']['duration']))
  popwts = plotMeanWeights(pdf,f_ax3,xl=(0,simConfig['simConfig']['duration']))
  plotHitMiss(actreward,f_ax4)
  possible_src = ['EV1', 'EV4', 'EMT','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE']
  lsrc = []
  for c_src in possible_src:
    if c_src in dstartidx:
      lsrc.append(c_src)
  print('Source Pops: ', lsrc)
  possible_targs = ['EMDOWN', 'EMUP','EMSTAY']
  ltarg = []
  for c_targ in possible_targs:
    if c_targ in dstartidx:
      ltarg.append(c_targ)  
  ltitle = []
  for src in lsrc:
    for trg in ltarg: ltitle.append(src+'->'+trg)
  dimg = {}; dline = {}; 
  def getwts (tdx, src):
    t = utimes[tdx]
    if 'EMSTAY' in dstartidx:
      ltarg = ['EMDOWN', 'EMUP','EMSTAY']
    else:
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
  if 'EMSTAY' in dstartidx:
    minW,maxW = np.min([np.min(popwts['EMDOWN']),np.min(popwts['EMUP']),np.min(popwts['EMSTAY'])]), np.max([np.max(popwts['EMDOWN']),np.max(popwts['EMUP']),np.max(popwts['EMSTAY'])])
  else:
    minW,maxW = np.min([np.min(popwts['EMDOWN']),np.min(popwts['EMUP'])]), np.max([np.max(popwts['EMDOWN']),np.max(popwts['EMUP'])])
  t = utimes[0]
  dline[1], = f_ax1.plot([t,t],[minR,maxR],'r',linewidth=0.2); f_ax1.set_xticks([])
  dline[2], = f_ax2.plot([t,t],[minW,maxW],'r',linewidth=0.2); f_ax2.set_xticks([])  
  pinds = 0
  fig.suptitle('Time=' + str(round(t,2)) + ' ms')
  for src in lsrc:
    wtsDOWN, wtsUP = getwts(0, src)
    ax=f_ax[pinds]
    dimg[pinds] = ax.imshow(wtsDOWN, origin='upper', cmap=cmap, vmin=minwt, vmax=maxwt+wtrange)
    ax.set_title(ltitle[pinds]); ax.axis('off')
    pinds+=1
    ax=f_ax[pinds]
    dimg[pinds] = ax.imshow(wtsUP, origin='upper', cmap=cmap, vmin=minwt, vmax=maxwt+wtrange)
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
      wtsDOWN, wtsUP = getwts(tdx, src)
      dimg[pinds].set_data(wtsDOWN)
      pinds+=1
      dimg[pinds].set_data(wtsUP)
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
  possible_src = ['EV1', 'EV4', 'EMT','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE']
  lsrc = []
  for c_src in possible_src:
    if c_src in dstartidx:
      lsrc.append(c_src)
  print('Source Pops: ', lsrc)
  possible_targs = ['EMDOWN', 'EMUP','EMSTAY']
  ltrg = []
  for c_targ in possible_targs:
    if c_targ in dstartidx:
      ltrg.append(c_targ)
  for src in lsrc:
      for trg in ltrg:
          davgw[src+'->'+trg] = arr = []        
          for t in utimes:
              pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[trg]) & (pdf.postid<=dendidx[trg]) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
              arr.append(np.mean(pdfs.weight))
      subplot(12,1,gdx)
      plot(utimes,davgw[src+'->EMDOWN'],'r-',linewidth=3);
      plot(utimes,davgw[src+'->EMUP'],'b-',linewidth=3);
      if 'EMSTAY' in dstartidx:
        plot(utimes,davgw[src+'->EMSTAY'],'g-',linewidth=3);
        legend((src+'->EMDOWN',src+'->EMUP',src+'->EMSTAY'),loc='upper left')
      else:
        legend((src+'->EMDOWN',src+'->EMUP'),loc='upper left')
      plot(utimes,davgw[src+'->EMDOWN'],'ro',markersize=10);
      plot(utimes,davgw[src+'->EMUP'],'bo',markersize=10);
      if 'EMSTAY' in dstartidx:
        plot(utimes,davgw[src+'->EMSTAY'],'go',markersize=10);       
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
    subplot(4,1,1)
    plot(actreward.time,actreward.reward,'k',linewidth=4)
    plot(actreward.time,actreward.reward,'ko',markersize=10)  
    xlim((0,simConfig['simConfig']['duration']))
    ylim((-1.1,1.1))
    ylabel('critic')
    title('sum of weights on to post-synaptic neurons')
    for trg in ['EMDOWN', 'EMUP','EMSTAY']:
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
    subplot(4,1,2)
    plot(utimes,np.array(wperPostID[src+'->EMDOWN']),'r-o',linewidth=3,markersize=5)
    #legend((src+'->EMDOWN'),loc='upper left')
    xlim((0,simConfig['simConfig']['duration']))
    ylabel(src+'->EMDOWN weights')
    subplot(4,1,3)
    plot(utimes,np.array(wperPostID[src+'->EMUP']),'b-o',linewidth=3,markersize=5) 
    #legend((src+'->EMUP'),loc='upper left')       
    xlim((0,simConfig['simConfig']['duration']))
    ylabel(src+'->EMUP weights')
    subplot(4,1,4)
    plot(utimes,np.array(wperPostID[src+'->EMSTAY']),'g-o',linewidth=3,markersize=5) 
    #legend((src+'->EMUP'),loc='upper left')       
    xlim((0,simConfig['simConfig']['duration']))
    ylabel(src+'->EMSTAY weights') 
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
    subplot(4,1,1)
    plot(actreward.time,actreward.reward,'k',linewidth=4)
    plot(actreward.time,actreward.reward,'ko',markersize=10)  
    xlim((0,simConfig['simConfig']['duration']))
    ylim((-1.1,1.1))
    ylabel('critic')
    colorbar
    title('sum of weights on to post-synaptic neurons')
    for trg in ['EMDOWN', 'EMUP','EMSTAY']:
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
    subplot(4,1,2)
    imshow(np.transpose(np.array(wperPostID[src+'->EMDOWN'])),aspect = 'auto',cmap='hot', interpolation='None')
    b1 = gca().get_xticks()
    gca().set_xticks(b1-1)
    gca().set_xticklabels((100*b1).astype(int))
    colorbar(orientation='horizontal',fraction=0.05)
    #legend((src+'->EMDOWN'),loc='upper left')
    xlim((-1,b1[-1]-1))
    ylabel(src+'->EMDOWN weights')
    subplot(4,1,3)
    imshow(np.transpose(np.array(wperPostID[src+'->EMUP'])),aspect = 'auto',cmap='hot', interpolation='None') 
    b2 = gca().get_xticks()
    gca().set_xticks(b2-1)
    gca().set_xticklabels((100*b2).astype(int))
    colorbar(orientation='horizontal',fraction=0.05)
    #legend((src+'->EMUP'),loc='upper left')       
    xlim((-1,b2[-1]-1))
    ylabel(src+'->EMUP weights') 
    xlabel('Time (ms)')
    subplot(4,1,4)
    imshow(np.transpose(np.array(wperPostID[src+'->EMSTAY'])),aspect = 'auto',cmap='hot', interpolation='None') 
    b2 = gca().get_xticks()
    gca().set_xticks(b2-1)
    gca().set_xticklabels((100*b2).astype(int))
    colorbar(orientation='horizontal',fraction=0.05)
    #legend((src+'->EMUP'),loc='upper left')       
    xlim((-1,b2[-1]-1))
    ylabel(src+'->EMSTAY weights') 
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
    for trg in ['EMDOWN','EMUP','EMSTAY']:
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
      pdfsDOWN = pdf[(pdf.time==t) & (pdf.postid==targetMDOWN_postID) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
      pdfdsUP = pdf[(pdf.time==t) & (pdf.postid==targetMUP_postID) & (pdf.preid>=dstartidx[src]) & (pdf.preid<=dendidx[src])]
      upreLCells = np.unique(pdfsDOWN.preid)
      upreRCells = np.unique(pdfdsUP.preid)
      for preID in upreLCells:
        pdfs1 = pdfsDOWN[(pdfsDOWN.preid==preID)]
        p1 = np.array(pdfs1.weight) #may have more than 1 weight---as two cells may have both AMPA and NMDA syns
        for w in p1:
          arrL[tstep].append(w)
          arrL2[tstep].append(preID)
      for preID in upreRCells:
        pdfs2 = pdfdsUP[(pdfdsUP.preid==preID)]
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

#
def getinputmap (pdf, t, prety, postid, poty, dnumc, dstartidx, dendidx, asweight=False):
  nrow = ncol = int(np.sqrt(dnumc[prety]))
  rfmap = np.zeros((nrow,ncol))
  pdfs = pdf[(pdf.postid==postid) & (pdf.preid>=dstartidx[prety]) & (pdf.preid<=dendidx[prety]) & (pdf.time==t)]
  if len(pdfs) < 1: return rfmap
  if not asweight:
    for idx in pdfs.index:
      preid = int(pdfs.at[idx,'preid'])
      x,y = gid2pos(dnumc[prety], dstartidx[prety], preid)
      rfmap[y,x] += 1
  else:
    rfcnt = np.zeros((nrow,ncol))    
    for idx in pdfs.index:
      preid = int(pdfs.at[idx,'preid'])
      x,y = gid2pos(dnumc[prety], dstartidx[prety], preid)
      rfcnt[y,x] += 1
      rfmap[y,x] += pdfs.at[idx,'weight']
    for y in range(nrow):
      for x in range(ncol):
        if rfcnt[y,x]>0: rfmap[y,x]/=rfcnt[y,x]
  return rfmap

#
def getallinputmaps (pdf, t, postid, poty, dnumc, dstartidx, dendidx, lprety = ['EV1DNW', 'EV1DN', 'EV1DNE', 'EV1DW', 'EV1','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE'], asweight=False):
  # gets all input maps onto postid
  return {prety:getinputmap(pdf, t, prety, postid, poty, dnumc, dstartidx, dendidx, asweight=asweight) for prety in lprety}

#
def plotallinputmaps (pdf, t, postid, poty, dnumc, dstartidx, dendidx, lprety=['EV1DNW', 'EV1DN', 'EV1DNE', 'EV1DW', 'EV1','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE'], asweight=False, cmap='jet',dmap=None):
  if dmap is None:
    drfmap = getallinputmaps(pdf, t, postid, poty, dnumc, dstartidx, dendidx, lprety, asweight=asweight)
  else:
    drfmap = dmap
  vmin,vmax = 1e9,-1e9
  for prety in lprety:
    vmin = min(vmin, np.amin(drfmap[prety]))
    vmax = max(vmax, np.amax(drfmap[prety]))    
  for tdx,prety in enumerate(lprety):
    subplot(3,3,tdx+1)
    imshow(drfmap[prety],cmap=cmap,origin='upper',vmin=vmin,vmax=vmax);
    title(prety+'->'+poty+str(postid));
    colorbar()
  return drfmap
  
 #
def getrecurrentmap(pdf, t, nety, dnumc, dstartidx, dendidx, asweight=False): #Individual map for pop
  postid = dstartidx[nety] + 0  #checks reccurrent connectivity to postid neuron, which is first neuron in pop
  nrow = ncol = int(np.sqrt(dnumc[nety]))
  rfmap = np.zeros((nrow,ncol))
  pdfs = pdf[(pdf.postid==postid) & (pdf.preid>dstartidx[nety]) & (pdf.preid<=dendidx[nety]) & (pdf.time==t)]
  if len(pdfs) < 1: return rfmap
  if not asweight:
    for idx in pdfs.index:
      preid = int(pdfs.at[idx,'preid'])
      x,y = gid2pos(dnumc[nety], dstartidx[nety], preid)
      rfmap[y,x] += 1
  else:
    rfcnt = np.zeros((nrow,ncol))
    for idx in pdfs.index:
      preid = int(pdfs.at[idx,'preid'])
      x,y = gid2pos(dnumc[nety], dstartidx[nety], preid)
      rfcnt[y,x] += 1
      rfmap[y,x] += pdfs.at[idx,'weight']
    for y in range(nrow):
      for x in range(ncol):
        if rfcnt[y,x]>0: rfmap[y,x]/=rfcnt[y,x]         #rfmap integrates weight, take the average
  return rfmap

 #
def getallrecurrentmaps (pdf, t, dnumc, dstartidx, dendidx, lnety = ['EV1DNW', 'EV1DN', 'EV1DNE', 'EV1DW', 'EV1','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE'], asweight=False):
  # gets all recurrent maps in lnety
  return {nety:getrecurrentmap(pdf, t, nety, dnumc, dstartidx, dendidx, asweight=asweight) for nety in lnety}

 #
def plotallrecurrentmaps (pdf, t, dnumc, dstartidx, dendidx, lnety = ['EV1DNW', 'EV1DN', 'EV1DNE', 'EV1DW', 'EV1','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE'], asweight=False, cmap='jet',dmap=None):
  if dmap is None:
    drfmap = getallrecurrentmaps(pdf, t, dnumc, dstartidx, dendidx, lnety, asweight=asweight)
  else:
    drfmap = dmap
  vmin,vmax = 1e9,-1e9
  for nety in lnety:
    vmin = min(vmin, np.amin(drfmap[nety]))
    vmax = max(vmax, np.amax(drfmap[nety]))
  for tdx,nety in enumerate(lnety):
    postid = dstartidx[nety] + 0        #recalculate postid, same as in getrecurrentmap
    subplot(3,3,tdx+1)                  #3x3 plot, max9 subplots. Can be changed
    imshow(drfmap[nety],cmap=cmap,origin='upper',vmin=vmin,vmax=vmax);
    title(nety+'->'+nety+str(postid));
    colorbar()
  return drfmap

def getpopinputmap (pdf, t, dnumc, dstartidx, dendidx, poty, asweight=True):
  # get integrated RF for a population
  pdfs = pdf[(pdf.time==t) & (pdf.postid>=dstartidx[poty]) & (pdf.postid<=dendidx[poty])]
  ddrfmap = {}
  for idx in range(dstartidx[poty],dendidx[poty]+1,1):
    print(idx)
    ddrfmap[idx] = getallinputmaps(pdfs, np.amax(pdfs.time), idx, poty, dnumc, dstartidx, dendidx, asweight=asweight)
  dout = {}
  for idx in range(dstartidx[poty],dendidx[poty]+1,1):
    drfmap = ddrfmap[idx]
    for k in drfmap.keys():
      if k not in dout:
        dout[k] = drfmap[k]
      else:
        dout[k] += drfmap[k]
  return dout  

def analyzeRepeatedInputSequences(dact,dnumc,InputImages, targetPixel=(10,10),nbseq=14,targetCorr=0.9):
  midInds = np.where(InputImages[:,targetPixel[0],targetPixel[1]]>250)
  # for each midInd, find 14 (13 could be enough but i am not sure) consecutive Images to see the trajectory. 
  seqInputs = np.zeros((int(len(midInds[0])/2),nbseq,InputImages.shape[0],InputImages.shape[1]),dtype=float)
  seqActions = np.zeros((int(len(midInds[0])/2),nbseq),dtype=float)
  seqPropActions = np.zeros((int(len(midInds[0])/2),nbseq),dtype=float)
  seqRewards = np.zeros((int(len(midInds[0])/2),nbseq),dtype=float)
  seqHitMiss = np.zeros((int(len(midInds[0])/2),nbseq),dtype=float)
  seqOutputsUP = np.zeros((int(len(midInds[0])/2),nbseq,dact['EMUP'].shape[1],dact['EMUP'].shape[2]),dtype=float)
  seqOutputsDOWN = np.zeros((int(len(midInds[0])/2),nbseq,dact['EMDOWN'].shape[1],dact['EMDOWN'].shape[2]),dtype=float)
  if dnumc['EMSTAY']>0:
    seqOutputsSTAY = np.zeros((int(len(midInds[0])/2),nbseq,dact['EMSTAY'].shape[1],dact['EMSTAY'].shape[2]),dtype=float)
  count = 0
  for i in range(0,len(midInds[0]),2):
    cmidInd = midInds[0][i]
    for j in range(nbseq):
      seqInputs[count,j,:,:] = InputImages[cmidInd+j,:,:]
      seqActions[count,j] = actreward['action'][cmidInd+j]
      seqRewards[count,j] = actreward['reward'][cmidInd+j]
      seqPropActions[count,j] = actreward['proposed'][cmidInd+j]
      seqHitMiss[count,j] = actreward['hit'][cmidInd+j]
      seqOutputsUP[count,j,:,:] = dact['EMUP'][cmidInd+j,:,:]
      seqOutputsDOWN[count,j,:,:] = dact['EMDOWN'][cmidInd+j,:,:]
      if dnumc['EMSTAY']>0:
        seqOutputsSTAY[count,j,:,:] = dact['EMSTAY'][cmidInd+j,:,:]
    count = count + 1
  # now i have all inputs, outputs, actions and proposed etc for all inputs where the ball starts in the middle of the screen.
  # But i need to pick up the sequences which are exactly like one another.  
  x = np.sum(seqInputs,axis=1)[1,:,:] #3:17
  goodInds = []
  for j in range(seqInputs.shape[0]):
    y = np.sum(seqInputs,axis=1)[j,:,:]
    corr, p_value = pearsonr(x.flat, y.flat)
    if corr>targetCorr:
      goodInds.append(j)
  # for comparison only use correlated sequences...
  seqInputs4comp = seqInputs[goodInds,:,:,:]
  seqActions4comp = seqActions[goodInds,:]
  seqRewards4comp = seqRewards[goodInds,:]
  seqPropActions4comp = seqPropActions[goodInds,:]
  seqHitMiss4comp = seqHitMiss[goodInds,:]
  seqOutputsUP4comp = seqOutputsUP[goodInds,:,:,:]
  seqOutputsDOWN4comp = seqOutputsDOWN[goodInds,:,:,:]
  if dnumc['EMSTAY']>0:
    seqOutputsSTAY4comp = seqOutputsSTAY[goodInds,:,:,:]
  lSeqNBs4comp = [0,1,2,3,4,5,6,7,8,9,10]
  fig, axs = plt.subplots(6, 5, figsize=(10,8));
  lax = axs.ravel()
  for i in range(5):
    cSeq = lSeqNBs4comp[i]
    lax[i].imshow(np.sum(seqInputs4comp,axis=1)[cSeq])
    lax[i].axis('off')
    lax[i+5].plot(np.sum(np.sum(seqOutputsUP4comp,axis=2),axis=2)[cSeq],'b-o',markersize=3)
    lax[i+5].plot(np.sum(np.sum(seqOutputsDOWN4comp,axis=2),axis=2)[cSeq],'r-o',markersize=3)
    if dnumc['EMSTAY']>0:
      lax[i+5].plot(np.sum(np.sum(seqOutputsSTAY4comp,axis=2),axis=2)[cSeq],'g-o',markersize=3)
    if i==0: lax[i+5].set_ylabel('# of pop spikes')
    lax[i+10].plot(seqActions4comp[cSeq,:],'-o',color=(0,0,0,1),markersize=3)
    lax[i+10].plot(seqPropActions[cSeq,:],'-o',color=(0.5,0.5,0.5,1),markersize=3)
    lax[i+10].set_yticks([1,3,4])
    if i==0: lax[i+10].set_yticklabels(['STAY','DOWN','UP'])
    cSeq = lSeqNBs4comp[i+5]
    lax[i+15].imshow(np.sum(seqInputs4comp,axis=1)[cSeq])
    lax[i+15].axis('off')
    lax[i+20].plot(np.sum(np.sum(seqOutputsUP4comp,axis=2),axis=2)[cSeq],'b-o',markersize=3)
    lax[i+20].plot(np.sum(np.sum(seqOutputsDOWN4comp,axis=2),axis=2)[cSeq],'r-o',markersize=3)
    if dnumc['EMSTAY']>0:
      lax[i+20].plot(np.sum(np.sum(seqOutputsSTAY4comp,axis=2),axis=2)[cSeq],'g-o',markersize=3)
    if i==0: lax[i+20].set_ylabel('# of pop spikes')
    lax[i+25].plot(seqActions4comp[cSeq,:],'-o',color=(0,0,0,1),markersize=3)
    lax[i+25].plot(seqPropActions[cSeq,:],'-o',color=(0.5,0.5,0.5,1),markersize=3)
    lax[i+25].set_yticks([1,3,4])
    if i==0: lax[i+25].set_yticklabels(['STAY','DOWN','UP'])
  if dnumc['EMSTAY']>0:
    lax[i+5].legend(['UP','DOWN','STAY'],loc='best')
  else:
    lax[i+5].legend(['UP','DOWN'],loc='best')
  lax[i+10].legend(['Actions','Proposed'],loc='best')


def analyzeRepeatedInputForSingleEvent(dact, dnumc, InputImages, targetPixel=(10,10)):
  midInds = np.where(InputImages[:,targetPixel[0],targetPixel[1]]>250)
  # for each midInd, find 14 (13 could be enough but i am not sure) consecutive Images to see the trajectory. 
  seqInputs = np.zeros((int(len(midInds[0])/2),20,20),dtype=float)
  seqActions = np.zeros((int(len(midInds[0])/2)),dtype=float)
  seqPropActions = np.zeros((int(len(midInds[0])/2)),dtype=float)
  seqRewards = np.zeros((int(len(midInds[0])/2)),dtype=float)
  seqHitMiss = np.zeros((int(len(midInds[0])/2)),dtype=float)
  seqOutputsUP = np.zeros((int(len(midInds[0])/2),dact['EMUP'].shape[1],dact['EMUP'].shape[2]),dtype=float)
  seqOutputsDOWN = np.zeros((int(len(midInds[0])/2),dact['EMDOWN'].shape[1],dact['EMDOWN'].shape[2]),dtype=float)
  if dnumc['EMSTAY']>0:
    seqOutputsSTAY = np.zeros((int(len(midInds[0])/2),dact['EMSTAY'].shape[1],dact['EMSTAY'].shape[2]),dtype=float)
  count = 0
  for i in range(0,len(midInds[0]),2):
    cmidInd = midInds[0][i]
    seqInputs[count,:,:] = InputImages[cmidInd,:,:]
    seqActions[count] = actreward['action'][cmidInd]
    seqRewards[count] = actreward['reward'][cmidInd]
    seqPropActions[count] = actreward['proposed'][cmidInd]
    seqHitMiss[count] = actreward['hit'][cmidInd]
    seqOutputsUP[count,:,:] = dact['EMUP'][cmidInd,:,:]
    seqOutputsDOWN[count,:,:] = dact['EMDOWN'][cmidInd,:,:]
    if dnumc['EMSTAY']>0:
      seqOutputsSTAY[count,:,:] = dact['EMSTAY'][cmidInd,:,:]
    count = count + 1
  FR_UP = np.sum(np.sum(seqOutputsUP,axis=1),axis=1)
  FR_DOWN = np.sum(np.sum(seqOutputsDOWN,axis=1),axis=1)
  if dnumc['EMSTAY']>0:
    FR_STAY = np.sum(np.sum(seqOutputsSTAY,axis=1),axis=1)
  fig, axs = plt.subplots(2, 3, figsize=(12,7));
  lax = axs.ravel()
  lax[0].imshow(np.sum(seqInputs,axis=0))
  lax[0].axis('off')
  lax[1].hist(seqActions,bins=[-1.5,-0.5,0.5,1.5,2.5,3.5,4.5])
  lax[1].set_xlabel('Actions')
  lax[1].set_xticks([1,3,4])
  lax[1].set_xticklabels(['STAY','DOWN','UP'])
  lax[2].hist(seqPropActions,bins=[-1.5,-0.5,0.5,1.5,2.5,3.5,4.5])
  lax[2].set_xlabel('Proposed Actions')
  lax[2].set_xticks([1,3,4])
  lax[2].set_xticklabels(['STAY','DOWN','UP'])
  lax[3].hist(FR_UP)
  lax[3].set_xlabel('# of EMUP spikes')
  lax[4].hist(FR_DOWN)
  lax[4].set_xlabel('# of EMDOWN spikes')
  if dnumc['EMSTAY']>0:
    lax[5].hist(FR_STAY)
    lax[5].set_xlabel('# of EMSTAY spikes')
  #
  fig, axs = plt.subplots(4, 1, figsize=(10,8));
  lax = axs.ravel()
  lax[0].imshow(np.sum(seqInputs,axis=0))
  lax[0].axis('off')
  lax[1].plot(np.sum(np.sum(seqOutputsUP,axis=1),axis=1),'b-o',markersize=3)
  lax[1].plot(np.sum(np.sum(seqOutputsDOWN,axis=1),axis=1),'r-o',markersize=3)
  if dnumc['EMSTAY']>0:
    lax[1].plot(np.sum(np.sum(seqOutputsSTAY,axis=1),axis=1),'g-o',markersize=3)
  lax[1].set_ylabel('# of pop spikes')
  lax[2].plot(seqActions,'-o',color=(0,0,0,1),markersize=3)
  lax[2].plot(seqPropActions,'-o',color=(0.5,0.5,0.5,1),markersize=3)
  lax[2].set_yticks([1,3,4])
  lax[2].set_yticklabels(['STAY','DOWN','UP'])
  lax[3].plot(seqRewards ,'-o',color=(0,0,0,1),markersize=3)
  lax[3].plot(seqHitMiss,'-o',color=(0.5,0.5,0.5,1),markersize=3)
  lax[3].set_yticks([-1,0,1])
  lax[3].legend(['Rewards','Hit/Moss'])
  if dnumc['EMSTAY']>0:
    lax[1].legend(['UP','DOWN','STAY'],loc='best')
  else:
    lax[1].legend(['UP','DOWN'],loc='best')
  lax[2].legend(['Actions','Proposed'],loc='best')

"""
current_time_stepNB = 0
cumRewardActions = []
cumPunishingActions = []
f_ax = []
fig = []
def updateBehaviorPlot (sim,InputImages,Images,dirSensitiveNeurons,Racket_pos,Ball_pos, current_time_stepNB,f_ax,fig):
  # update 
  global cumRewardActions, cumPunishingActions
  maxtstr = len(str(100000))
  if current_time_stepNB==0:
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(4,4)
    f_ax = []
    f_ax.append(fig.add_subplot(gs[0:2,0])) #for 5-image input - 0
    f_ax.append(fig.add_subplot(gs[0:2,1])) #for single image  - 1
    f_ax.append(fig.add_subplot(gs[0:2,2])) #for direction selectivity - 2
    f_ax.append(fig.add_subplot(gs[2,0:2])) #display executed/proposed actions - 3
    f_ax.append(fig.add_subplot(gs[2,2:4])) #display - 4 
    f_ax.append(fig.add_subplot(gs[3,0:2])) #- 5
    f_ax.append(fig.add_subplot(gs[3,2:4])) #- 6
  cbaxes = fig.add_axes([0.75, 0.62, 0.01, 0.24])
  f_ax[0].cla()
  f_ax[0].imshow(InputImages[-1])
  f_ax[0].set_title('Input Images [t-5,t]')
  f_ax[2].cla()
  fa = f_ax[2].imshow(dirSensitiveNeurons,origin='upper',vmin=0, vmax=359, cmap='Dark2')
  f_ax[2].set_xlim((-0.5,9.5))
  f_ax[2].set_ylim((9.5,-0.5))
  f_ax[2].set_xticks(ticks=[0,2,4,6,8])
  f_ax[2].set_title('direction angles [t-5,t]')
  c1 = plt.colorbar(fa,cax = cbaxes)
  c1.set_ticks([22,67,112,157,202,247,292,337])
  c1.set_ticklabels(['E','NE','N','NW','W','SW','S','SE'])
  Hit_Missed = np.array(sim.allHits)
  allHit = np.where(Hit_Missed==1,1,0) 
  allMissed = np.where(Hit_Missed==-1,1,0)
  cumHits = np.cumsum(allHit) #cummulative hits evolving with time.
  cumMissHits = np.cumsum(allMissed) #if a reward is -1, replace it with 1 else replace it with 0.
  Diff_Actions_Proposed = np.subtract(sim.allActions,sim.allProposedActions)
  t0 = int(dconf['actionsPerPlay'])
  tpnts = range(t0,len(Diff_Actions_Proposed)+t0,t0)
  rewardingActions = np.sum(np.where(Diff_Actions_Proposed==0,1,0))
  punishingActions = np.sum(np.where((Diff_Actions_Proposed>0) | (Diff_Actions_Proposed<0),1,0))
  totalActs = rewardingActions + punishingActions
  cumRewardActions.append(rewardingActions/totalActs)
  cumPunishingActions.append(punishingActions/totalActs)
  f_ax[3].plot(sim.allActions,LineStyle="None",Marker=2,MarkerSize=6,MarkerFaceColor="None",MarkerEdgeColor='r')
  f_ax[3].plot(sim.allProposedActions,LineStyle="None",Marker=3,MarkerSize=6,MarkerFaceColor="None",MarkerEdgeColor='b')
  f_ax[3].set_yticks(ticks=[1,3,4])
  f_ax[3].set_yticklabels(labels=['No action','Down','Up'])
  f_ax[3].set_ylim((0.5,4.5))
  f_ax[3].legend(('Executed','Proposed'),loc='upper left')
  f_ax[4].cla()
  f_ax[4].plot(tpnts,np.array(cumRewardActions),'o-',MarkerSize=5,MarkerFaceColor='r',MarkerEdgeColor='r')
  f_ax[4].plot(tpnts,np.array(cumPunishingActions),'s-',MarkerSize=5,MarkerFaceColor='b',MarkerEdgeColor='b')
  f_ax[4].legend(('Rewarding actions','Punishing Actions'),loc='upper left')
  f_ax[5].cla()
  f_ax[5].plot(sim.allRewards,'o-',MarkerFaceColor="None",MarkerEdgeColor='g')
  f_ax[5].legend('Rewards')
  f_ax[6].cla()
  f_ax[6].plot(cumHits,Marker='o',MarkerSize=5,MarkerFaceColor='r',MarkerEdgeColor='r')
  f_ax[6].plot(cumMissHits,Marker='s',MarkerSize=3,MarkerFaceColor='k',MarkerEdgeColor='k')
  f_ax[6].legend(('Cumm. Hits','Cumm. Miss'),loc='upper left')
  f_ax[1].cla()
  for nbi in range(np.shape(Racket_pos)[0]):
    f_ax[1].imshow(Images[nbi])
    if Ball_pos[nbi][0]>18: #to account for offset for the court
      f_ax[1].plot(Racket_pos[nbi][0],Racket_pos[nbi][1],'o',MarkerSize=5, MarkerFaceColor="None",MarkerEdgeColor='r')
      f_ax[1].plot(Ball_pos[nbi][0],Ball_pos[nbi][1],'o',MarkerSize=5, MarkerFaceColor="None",MarkeredgeColor='b')
    f_ax[1].set_title('last obs')
    #plt.pause(0.1)
    ctstrl = len(str(current_time_stepNB))
    tpre = ''
    for ttt in range(maxtstr-ctstrl):
      tpre = tpre+'0'
    fn = tpre+str(current_time_stepNB)+'.png'
    fnimg = '/tmp/'+fn
    plt.savefig(fnimg)
    #plt.close() 
    #lfnimage.append(fnimg)
    current_time_stepNB = current_time_stepNB+1
  return current_time_stepNB, f_ax, fig
"""

if __name__ == '__main__':
  stepNB = -1
  if len(sys.argv) > 1:
    try:
      stepNB = int(sys.argv[1]) #which file(stepNB) want to plot
    except:
      pass
  print(stepNB)
  allpossible_pops = ['ER','IR','EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE','IV1','EV4','IV4','EMT','IMT','EMDOWN','EMUP','EMSTAY','IM']
  lpop = []
  for pop_ind in range(len(allpossible_pops)):
    cpop = allpossible_pops[pop_ind]
    #print('cpop',cpop)
    if cpop in list(dconf['net']['allpops'].keys()):
      if dconf['net']['allpops'][cpop]>0:
        lpop.append(cpop)
  print('lpop: ', lpop)
  simConfig, pdf, actreward, dstartidx, dendidx, dnumc, dspkID, dspkT, InputImages, ldflow, dact = loadsimdat(getactmap=False,lpop=lpop)
  dstr = getdatestr(); simstr = dconf['sim']['name'] # date and sim string
  print('loaded simulation data',simstr,'on',dstr)
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
