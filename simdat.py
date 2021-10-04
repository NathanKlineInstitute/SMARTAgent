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
from mpl_toolkits.mplot3d import Axes3D
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
allpossible_pops = list(dconf['net']['allpops'].keys())

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
        A.append([row[0], preID, poID, row[1]]) # A.append([row[0], preID, poID, row[1], row[2]])
  return pd.DataFrame(A,columns=['time','preid','postid','weight']) # ,(['cumreward'])

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

def shuffleweights (pdf):
  # shuffle the weights
  npwt = np.array(pdf.weight)
  np.random.shuffle(npwt)
  Ashuf = np.array([pdf.time,pdf.preid,pdf.postid,npwt]).T
  pdfshuf = pd.DataFrame(Ashuf,columns=['time','preid','postid','weight'])
  return pdfshuf
  #D = pdf2weightsdict(pdfshuf);
  #return D

def getsimname (name=None):
  if name is None:
    if stepNB == -1: name = dconf['sim']['name']
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
  actreward=None
  try:
    actreward = pd.DataFrame(np.loadtxt('data/'+name+'ActionsRewards.txt'),columns=['time','action','reward','proposed','hit','followtargetsign'])
  except:
    pass
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
  try:
    InputImages = loadInputImages(dconf['sim']['name'])
  except:
    pass
  try:
    ldflow = loadMotionFields(dconf['sim']['name'])
  except:
    pass
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
                   ,'EV1DNE','EV1DE','EV1DSW','EV1DS', 'EV1DSE','EMDOWN','EMUP','IM','EA','IA','EA2','IA2']
  possible_titles = ['Excit R', 'Excit V1', 'Excit V4', 'Excit MT', 'Inhib R', 'Inhib V1', 'Inhib V4', 'Inhib MT',\
                     'W','NW','N','NE','E','SW','S','SE','Excit M DOWN', 'Excit M UP', 'Excit M STAY', 'Inhib M',\
                     'Excit A' , 'Inhib A', 'Excit A2', 'Inhib A2']
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
      flowrow,flowcol = int(np.sqrt(dnumc['EV1DE'])),int(np.sqrt(dnumc['EV1DE']))
      X, Y = np.meshgrid(np.arange(0, flowcol, 1), np.arange(0,flowrow,1))
      ddat[ldx] = ax.quiver(X,Y,ldflow[0]['flow'][:,:,0],-ldflow[0]['flow'][:,:,1], pivot='mid', units='inches',width=0.022,scale=1/0.15)
      ax.set_xlim((0,flowcol)); ax.set_ylim((0,flowrow))
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
        ddat[ldx].set_UVC(ldflow[t+offidx]['flow'][:,:,0],-ldflow[t+offidx]['flow'][:,:,1])        
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

def viewInput (t, InputImages, ldflow, dhist, lpop = None, lclr = ['r','b'], twin=100, dobjpos = None):
  ax = subplot(2,2,1)
  tdx = int(t / tstepPerAction)
  imshow( InputImages[tdx][:,:], origin='upper', cmap='gray'); colorbar()
  ax.set_title('t = ' + str(t))
  objfctr = 1.0/8
  if dconf['DirectionDetectionAlgo']['UseFull']: objfctr=1/8.
  minobjt = dobjpos['time'][0]
  objofftdx = int(minobjt/tstepPerAction)
  def drobjpos (ax, tdx):  
    if dobjpos is not None and tdx - objofftdx >= 0:
      lobjx,lobjy = [objfctr*dobjpos[k][tdx-objofftdx][0] for k in ['ball','racket']], [objfctr*dobjpos[k][tdx-objofftdx][1] for k in ['ball','racket']]
      #print('lobjx:',lobjx,'lobjy:',lobjy)
      for k in ['ball','racket']:
        print('t=',tstepPerAction*(tdx),k,'x=',objfctr*dobjpos[k][tdx-objofftdx][0],'y=',objfctr*dobjpos[k][tdx-objofftdx][1])
      ax.plot(lobjx,lobjy,'ro')
  drobjpos(ax, tdx)
  ax = subplot(2,2,2)
  tdx += 1
  imshow( InputImages[tdx][:,:], origin='upper', cmap='gray'); colorbar();
  ax.set_title('t = ' + str(t+tstepPerAction))
  drobjpos(ax,tdx)
  tdx -= 1
  ax = subplot(2,2,3)
  ax.set_title('Motion, t = ' + str(t))
  X, Y = np.meshgrid(np.arange(0, InputImages[0].shape[1], 1), np.arange(0,InputImages[0].shape[0],1))
  ax.quiver(X,Y,ldflow[tdx]['flow'][:,:,0],-ldflow[tdx]['flow'][:,:,1], pivot='mid', units='inches',width=0.01,scale=1/0.9)
  ax.set_xlim((0,InputImages[0].shape[1])); ax.set_ylim((0,InputImages[0].shape[0]))
  ax.invert_yaxis()
  ax = subplot(2,2,4)
  for pop,clr in zip(lpop,lclr): ax.plot(dhist[pop][0],dhist[pop][1],clr)
  lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(lclr,lpop)]
  ax=gca(); ax.legend(handles=lpatch,handlelength=1)
  ax.set_xlim(tdx*tstepPerAction - twin, tdx*tstepPerAction + twin)
  xlabel('Time (ms)'); ylabel('Spikes')
  

#
def animInput (InputImages, outpath, framerate=50, figsize=None, showflow=False, ldflow=None, dobjpos=None,\
               actreward=None, nframe=None, skipopp=False):
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
  if dconf['net']['useBinaryImage']: lvmax = [1]
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
      # ax.set_ylabel(ltitle[idx])
      if dobjpos is not None:
        lobjx,lobjy = [objfctr*dobjpos[k][0,0] for k in dobjpos.keys()], [objfctr*dobjpos[k][0,1] for k in dobjpos.keys()]
        ddat['objpos'], = ax.plot(lobjx,lobjy,'ro')
    else:
      X, Y = np.meshgrid(np.arange(0, InputImages[0].shape[1], 1), np.arange(0,InputImages[0].shape[0],1))
      ddat[ldx] = ax.quiver(X,Y,ldflow[0]['flow'][:,:,0],-ldflow[0]['flow'][:,:,1], pivot='mid', units='inches',width=0.01,scale=1/0.3)
      ax.set_xlim((0,InputImages[0].shape[1])); ax.set_ylim((0,InputImages[0].shape[0]))
      ax.invert_yaxis()
    idx += 1
  cumHits, cumMissed, cumScore = None,None,None
  if actreward is not None:
    cumHits, cumMissed, cumScore = getCumPerfCols(actreward)    
  def updatefig (t):
    stitle = 'Time = ' + str(t*tstepPerAction) + ' ms'
    if cumHits is not None:
      if dconf['rewardcodes']['scorePoint'] > 0.0:      
        if skipopp:
          stitle += '\nModel Points:'+str(cumScore[t]) + '   Model Hits:'+str(cumHits[t])
        else:
          stitle += '\nOpponent Points:'+str(cumMissed[t])+'   Model Points:'+str(cumScore[t]) + '   Model Hits:'+str(cumHits[t])
      else:
        stitle += '\nModel Hits:'+str(cumHits[t]) + '   Model Misses:'+str(cumMissed[t])
    fig.suptitle(stitle)
    if t < 1: return fig # already rendered t=0 above
    print('frame t = ', str(t*tstepPerAction))    
    for ldx,ax in enumerate(lax):
      if ldx == 0:
        ddat[ldx].set_data(lact[0][t,:,:])
        if dobjpos is not None:
          lobjx,lobjy = [objfctr*dobjpos[k][t,0] for k in dobjpos.keys()], [objfctr*dobjpos[k][t,1] for k in dobjpos.keys()]
          ddat['objpos'].set_data(lobjx,lobjy)
      else:
        ddat[ldx].set_UVC(ldflow[t]['flow'][:,:,0],-ldflow[t]['flow'][:,:,1])        
    return fig
  t1 = range(0,totalDur,tstepPerAction)
  if nframe is None: nframe = len(t1)
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
          'EMDOWN','EMUP']  
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
      ddat[ldx] = ax.quiver(X,Y,ldflow[0]['flow'][:,:,0],-ldflow[0]['flow'][:,:,1], pivot='mid', units='inches',width=0.022,scale=1/0.15)
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
        ddat[ldx].set_UVC(ldflow[t+offidx]['flow'][:,:,0],-ldflow[t]['flow'][:,:,1])        
      else:
        ddat[ldx].set_UVC(maxdirX[t+offidx,:,:],maxdirY[t+offidx,:,:])
    return fig
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=len(t1)-1)
  writer = anim.getwriter(outpath, framerate=framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  ion()
  return fig, axs, plt

def loadInputImages (name=None):
  try:
    fn = 'data/'+getsimname(name)+'InputImages.npy'
    print('loading input images from', fn)
    return np.load(fn)
  except:
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

def ObjPos2pd (dobjpos):
  # convert object pos dictionary to pandas dataframe (for selection)
  ballX,ballY = dobjpos['ball'][:,0],dobjpos['ball'][:,1]
  racketX,racketY = dobjpos['racket'][:,0],dobjpos['racket'][:,1]
  if 'time' in dobjpos:
    time = dobjpos['time']
  else:
    time = np.linspace(0,totalDur,len(dobjpos['ball']))
  pdpos = pd.DataFrame(np.array([time, ballX, ballY, racketX, racketY]).T,columns=['time','ballX','ballY','racketX','racketY'])
  return pdpos

def getdistvstimecorr (pdpos, ballxmin=137, ballxmax=141, minN=2):
  # get distance vs time
  pdposs = pdpos[(pdpos.ballY>-1.0) & (pdpos.ballX>ballxmin) & (pdpos.ballX<ballxmax)]
  lbally = np.unique(pdposs.ballY)
  dout = {}
  lr,ly,lN,lpval = [],[],[],[]
  for y in lbally:
    dout[y] = {}
    pdposss = pdposs[(pdposs.ballY==y)]
    dist = np.sqrt((pdposss.ballY - pdposss.racketY)**2)
    #plot(pdposss.time, dist)
    dout[y]['time'] = pdposss.time
    dout[y]['dist'] = dist
    dout[y]['rackety'] = pdposss.racketY
    r,p=0,0
    if len(pdposss.time) > 1: r,p = pearsonr(pdposss.time, dist)
    if len(dist) >= minN:
      lpval.append(p)
      lr.append(r)
      ly.append(y)
      lN.append(len(dist))
  dout['lbally'] = ly
  dout['lr'] = lr
  dout['lpval'] = lpval
  dout['lN'] = lN
  return dout


def getspikehist (spkT, numc, binsz, tmax):
  tt = np.arange(0,tmax,binsz)
  nspk = [len(spkT[(spkT>=tstart) & (spkT<tstart+binsz)]) for tstart in tt]
  nspk = [1e3*x/(binsz*numc) for x in nspk]
  return tt,nspk

#
def getrate (dspkT,dspkID, pop, dnumc, tlim=None):
  # get average firing rate for the population, over entire simulation
  nspk = len(dspkT[pop])
  ncell = dnumc[pop]
  if tlim is not None:
    spkT = dspkT[pop]
    nspk = len(spkT[(spkT>=tlim[0])&(spkT<=tlim[1])])
    return 1e3*nspk/((tlim[1]-tlim[0])*ncell)
  else:  
    return 1e3*nspk/(totalDur*ncell)

def pravgrates (dspkT,dspkID,dnumc,tlim=None):
  # print average firing rates over simulation duration
  for pop in dspkT.keys(): print(pop,round(getrate(dspkT,dspkID,pop,dnumc,tlim=tlim),2),'Hz')

#
def drawraster (dspkT,dspkID,tlim=None,msz=2,skipstim=True):
  # draw raster (x-axis: time, y-axis: neuron ID)
  lpop=list(dspkT.keys()); lpop.reverse()
  lpop = [x for x in lpop if not skipstim or x.count('stim')==0]  
  csm=cm.ScalarMappable(cmap=cm.prism); csm.set_clim(0,len(dspkT.keys()))
  lclr = []
  for pdx,pop in enumerate(lpop):
    color = csm.to_rgba(pdx); lclr.append(color)
    plot(dspkT[pop],dspkID[pop],'o',color=color,markersize=msz)
  if tlim is not None:
    xlim(tlim)
  else:
    xlim((0,totalDur))
  xlabel('Time (ms)')
  #lclr.reverse(); 
  lpatch = [mpatches.Patch(color=c,label=s+' '+str(round(getrate(dspkT,dspkID,s,dnumc,tlim=tlim),2))+' Hz') for c,s in zip(lclr,lpop)]
  ax=gca()
  ax.legend(handles=lpatch,handlelength=1,loc='best')
  ylim((0,sum([dnumc[x] for x in lpop])))

#
def drawcellVm (simConfig, ldrawpop=None,tlim=None, lclr=None):
  csm=cm.ScalarMappable(cmap=cm.prism); csm.set_clim(0,len(dspkT.keys()))
  if tlim is not None:
    dt = simConfig['simData']['t'][1]-simConfig['simData']['t'][0]    
    sidx,eidx = int(0.5+tlim[0]/dt),int(0.5+tlim[1]/dt)
  dclr = OrderedDict(); lpop = []
  for kdx,k in enumerate(list(simConfig['simData']['V_soma'].keys())):  
    color = csm.to_rgba(kdx);
    if lclr is not None and kdx < len(lclr): color = lclr[kdx]
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
def plotFollowBall (actreward, ax=None,cumulative=True,msz=3,binsz=1e3,color='r',pun=False):
  # plot probability of model racket following target(predicted ball y intercept) vs time
  # when cumulative == True, plots cumulative probability; otherwise bins probabilities over binsz interval
  # not a good way to plot probabilities over time when uneven sampling - could resample to uniform intervals ...
  # for now cumulative == False is not plotted at all ... 
  global tstepPerAction
  if ax is None: ax = gca()
  ax.plot([0,np.amax(actreward.time)],[0.5,0.5],'--',color='gray')    
  allproposed = actreward[(actreward.proposed!=-1)] # only care about cases when can suggest a proposed action
  if dconf['useFollowMoveOutput']:
    val = 1
    if pun: val = -1
    rewardingActions = np.where(allproposed.followtargetsign==val,1,0)
  else:
    rewardingActions = np.where(allproposed.proposed-allproposed.action==0,1,0)
  if cumulative:
    rewardingActions = np.cumsum(rewardingActions) # cumulative of rewarding action
    cumActs = np.array(range(1,len(allproposed)+1))
    aout = np.divide(rewardingActions,cumActs)
    ax.plot(allproposed.time,aout,'.',color=color,markersize=msz)
  else:
    nbin = int(binsz / (np.array(actreward.time)[1]-np.array(actreward.time)[0]))
    aout = avgfollow = [mean(rewardingActions[sidx:sidx+nbin]) for sidx in arange(0,len(rewardingActions),nbin)]
    ax.plot(np.linspace(0,np.amax(actreward.time),len(avgfollow)), avgfollow, color=color,linewidth=msz)
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
    ax.plot(action_times,cumHits/cumMissed,'-o',color=lclr[0],markersize=msz)
    ax.set_xlim((0,np.max(action_times)))
    ax.set_ylabel('Hit/Miss ('+str(round(cumHits[-1]/cumMissed[-1],2))+')')
    return cumHits[-1]/cumMissed[-1]
  else:
    ax.plot(action_times,cumHits,'-o',color=lclr[0],markersize=msz)
    ax.plot(action_times,cumMissed,'-o',color=lclr[1],markersize=msz)
    ax.set_xlim((0,np.max(action_times)))
    ax.set_ylim((0,np.max([cumHits[-1],cumMissed[-1]])))    
    ax.set_ylabel('Hit Ball ('+str(cumHits[-1])+'), Miss Ball ('+str(cumMissed[-1])+')')
    return cumHits[-1],cumMissed[-1]

#
def plotHitMissRatioPerStep (lpda,ax=None,clr='k'):
  if ax is None: ax=gca()
  lhit,lmiss = [],[]
  for pda in lpda: 
    hit,miss = plotHitMiss(pda,asratio=False,asbin=False,ax=ax)
    lhit.append(hit); lmiss.append(miss)
  cla()
  lrat = np.array(lhit)/lmiss
  cla(); plot(lrat,clr,linewidth=4); plot(lrat,clr+'o',markersize=15)
  xlabel('Step',fontsize=35); ylabel('Hit/miss ratio',fontsize=35); xlim((0-.1,len(lhit)-1+.1))
  return lrat
  
  
#  
def plotScoreMiss (actreward,ax=None,msz=3,asratio=False,clr='r'):
  if ax is None: ax = gca()
  action_times = np.array(actreward.time)
  Hit_Missed = np.array(actreward.hit)
  allMissed = np.where(Hit_Missed==-1,1,0)
  cumMissed = np.cumsum(allMissed) #if a reward is -1, replace it with 1 else replace it with 0.
  cumScore = getCumScore(actreward)
  if asratio:
    ax.plot(action_times,cumScore/cumMissed,'-o',color=clr,markersize=msz)
    ax.set_xlim((0,np.max(action_times)))
    ax.set_ylabel('Score/Miss ('+str(round(cumScore[-1]/cumMissed[-1],2))+')')
    return cumScore[-1]/cumMissed[-1]
  else:
    ax.plot(action_times,cumScore,'-o',color=clr,markersize=msz)
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

def getCumPerfCols (actreward):
  # get cumulative performance arrays
  action_times = np.array(actreward.time)
  Hit_Missed = np.array(actreward.hit)
  allMissed = np.where(Hit_Missed==-1,1,0)
  cumMissed = np.cumsum(allMissed) #if a reward is -1, replace it with 1 else replace it with 0.  
  cumScore = getCumScore(actreward)
  #actreward['cumScoreRatio'] = cumScore/cumMissed # cumulative score/loss ratio
  allproposed = actreward[(actreward.proposed!=-1)] # only care about cases when can suggest a proposed action
  rewardingActions = np.where(allproposed.proposed-allproposed.action==0,1,0)
  rewardingActions = np.cumsum(rewardingActions) # cumulative of rewarding action
  cumActs = np.array(range(1,len(allproposed)+1))
  #actreward['cumFollow'] = np.divide(rewardingActions,cumActs) # cumulative follow probability
  allHit = np.where(Hit_Missed==1,1,0) 
  allMissed = np.where(Hit_Missed==-1,1,0)
  cumHits = np.cumsum(allHit) #cumulative hits evolving with time.
  cumMissed = np.cumsum(allMissed) #if a reward is -1, replace it with 1 else replace it with 0.
  #actreward['cumHitMissRatio'] = cumHits/cumMissed # cumulative hits/missed ratio
  return cumHits, cumMissed, cumScore
  

def plotPerf (actreward,yl=(0,1),asratio=True,asbin=False,binsz=10e3):
  # plot performance
  plotFollowBall(actreward,ax=subplot(1,1,1),cumulative=True,color='b');
  if dconf['useFollowMoveOutput']: plotFollowBall(actreward,ax=subplot(1,1,1),cumulative=True,color='m',pun=True);    
  plotHitMiss(actreward,ax=subplot(1,1,1),lclr=['g'],asratio=asratio,asbin=asbin,binsz=binsz); 
  plotScoreMiss(actreward,ax=subplot(1,1,1),clr='r',asratio=asratio);
  ylim(yl)
  ylabel('Performance')
  if dconf['useFollowMoveOutput']:
    lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(['b','m','g','r'],['Follow','Avoid','Hit/Miss','Score/Miss'])]
  else:
    lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(['b','g','r'],['Follow','Hit/Miss','Score/Miss'])]    
  ax=gca()
  ax.legend(handles=lpatch,handlelength=1)
  return ax

def plotComparePerf (lpda, lclr, yl=(0,.55), lleg=None, skipfollow=False, skipscore=False, asratio=True,asbin=False,binsz=10e3):
  # plot comparison of performance of list of action rewards dataframes in lpda
  # lclr is color to plot
  # lleg is optional legend
  ngraph=3
  if skipfollow: ngraph-=1
  if skipscore: ngraph-=1
  for pda,clr in zip(lpda,lclr):
    gdx=1
    if not skipfollow: plotFollowBall(pda,ax=subplot(1,ngraph,gdx),cumulative=True,color=clr); ylim(yl); gdx+=1
    plotHitMiss(pda,ax=subplot(1,ngraph,gdx),lclr=[clr],asratio=asratio,asbin=asbin,binsz=binsz); ylim(yl); gdx+=1
    if not skipscore: plotScoreMiss(pda,ax=subplot(1,ngraph,gdx),clr=clr,asratio=asratio); ylim(yl);
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
    dspk, uspk = dhist['EMDOWN'][1][i], dhist['EMUP'][1][i]
    if dspk > uspk:
      actsel.append(dconf['moves']['DOWN'])
    elif uspk > dspk:
      actsel.append(dconf['moves']['UP'])
    else: # dspk == uspk:
      actsel.append(dconf['moves']['NOMOVE'])
  return actsel  

def getconcatweightpdf (lfn,usefinal=False):
  # concatenate the weights together so can look at cumulative rewards,actions,etc.
  # lfn is a list of actionrewards filenames from the simulation
  pdf = None
  for fn in lfn:
    try:
      wtmp = readinweights(fn,final=usefinal) 
    except:
      try:
        wtmp = readinweights(fn,final=True)
      except:
        print('could not load weights from', fn)
    if pdf is None:
      pdf = wtmp
    else:
      wtmp.time += np.amax(pdf.time)
      pdf = pdf.append(wtmp)
  return pdf

def getconcatactionreward (lfn):
  # concatenate the actionreward data frames together so can look at cumulative rewards,actions,etc.
  # lfn is a list of actionrewards filenames from the simulation
  pda = None
  for fn in lfn:
    if not fn.endswith('ActionsRewards.txt'): fn = 'data/'+fn+'ActionsRewards.txt'
    acl = pd.DataFrame(np.loadtxt(fn),columns=['time','action','reward','proposed','hit','followtargetsign'])
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
    return [pd.DataFrame(np.loadtxt(fn),columns=['time','action','reward','proposed','hit','followtargetsign']) for fn in lfn]
  else:
    return [pd.DataFrame(np.loadtxt('data/'+fn+'ActionsRewards.txt'),columns=['time','action','reward','proposed','hit','followtargetsign']) for fn in lfn]    

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

def unnanl (l, val=0):
  out = []
  for x in l:
    if isnan(x):
      out.append(0)
    else:
      out.append(x)
  return out
      
def plotMeanWeights (pdf,ax=None,msz=1,xl=None,lpop=['EMDOWN','EMUP'],lclr=['k','r','b','g'],plotindiv=True,fsz=15,prety=None):
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
      if prety is not None:
        pdfs = pdfs[(pdfs.preid>=dstartidx[prety]) & (pdfs.preid<=dendidx[prety])]
      popwts[pop] = unnanl([np.mean(pdfs[(pdfs.time==t)].weight) for t in utimes]) #wts of connections onto pop      
      ax.plot(utimes,popwts[pop],clr+'-o',markersize=msz)
      mnw=min(mnw, np.amin(popwts[pop]))
      mxw=max(mxw, np.amax(popwts[pop]))            
  if xl is not None: ax.set_xlim(xl)
  ax.set_ylim((mnw,mxw))
  ax.set_ylabel('Average weight',fontsize=fsz);
  ax.set_xlabel('Time (ms)',fontsize=fsz);
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
  possible_targs = ['EMDOWN', 'EMUP']
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
  possible_targs = ['EMDOWN', 'EMUP']
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
    subplot(4,1,1)
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
  
def plotIndividualSynWeights(pdf):
  #plot 10% randomly selected connections
  utimes = np.unique(pdf.time)
  #for every postsynaptic neuron, find total weight of synaptic inputs per area (i.e. synaptic inputs from EV1, EV4 and EIT and treated separately for each cell——if there are 200 unique cells, will get 600 weights as 200 from each originating layer)
  allweights = {}
  preNeuronIDs = {}
  postNeuronIDs = {}
  #gdx = 2   
  for src in ['EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE', 'EV4', 'EMT']:
    for trg in ['EMDOWN','EMUP']:
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

def getinputconnmap (simConfig, prety, postid, dnumc, dstartidx, synMech, asweight=False):
  lcell = simConfig['net']['cells']
  cell = lcell[postid]
  nrow = ncol = int(np.sqrt(dnumc[prety]))
  cmap = np.zeros((nrow,ncol))
  for conn in cell['conns']:
    if lcell[conn['preGid']]['tags']['cellType'] == prety and conn['synMech']==synMech:
      x,y = gid2pos(dnumc[prety], dstartidx[prety], conn['preGid'])
      if asweight:
        cmap[y,x] = conn['weight']
      else:
        cmap[y,x] = 1
  return cmap

    
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

def analyzeRepeatedInputSequences(dact, InputImages, targetPixel=(10,10),nbseq=14,targetCorr=0.9):
  midInds = np.where(InputImages[:,targetPixel[0],targetPixel[1]]>250)
  repSeqInds = []
  for i in range(len(midInds[0])-1):
    if midInds[0][i+1]-midInds[0][i]<5:
      repSeqInds.append(i+1)
  uniqueSeqStartInds = []
  for i in range(len(midInds[0])):
    if i not in repSeqInds:
      uniqueSeqStartInds.append(midInds[0][i])
  # for each midInd, find 14 (13 could be enough but i am not sure) consecutive Images to see the trajectory.
  ImgH, ImgW = InputImages.shape[1],InputImages.shape[2]
  lmotorpop = [pop for pop in dconf['net']['EMotorPops'] if dconf['net']['allpops'][pop]>0] 
  seqInputs = np.zeros((len(uniqueSeqStartInds),nbseq,ImgH,ImgW),dtype=float)
  seqActions = np.zeros((len(uniqueSeqStartInds),nbseq),dtype=float)
  seqPropActions = np.zeros((len(uniqueSeqStartInds),nbseq),dtype=float)
  seqRewards = np.zeros((len(uniqueSeqStartInds),nbseq),dtype=float)
  seqHitMiss = np.zeros((len(uniqueSeqStartInds),nbseq),dtype=float)
  dseqOutputs = {pop:np.zeros((len(uniqueSeqStartInds),nbseq,dact[pop].shape[1],dact[pop].shape[2]),dtype=float) for pop in lmotorpop}
  for i in range(len(uniqueSeqStartInds)):
    cSeqStartInd = uniqueSeqStartInds[i]
    for j in range(nbseq):
      seqInputs[i,j,:,:] = InputImages[cSeqStartInd+j,:,:]
      seqActions[i,j] = actreward['action'][cSeqStartInd+j]
      seqRewards[i,j] = actreward['reward'][cSeqStartInd+j]
      seqPropActions[i,j] = actreward['proposed'][cSeqStartInd+j]
      seqHitMiss[i,j] = actreward['hit'][cSeqStartInd+j]
      for pop in dseqOutputs.keys():
        dseqOutputs[pop][i,j,:,:]=dact[pop][cSeqStartInd+j,:,:]
  # now i have all inputs, outputs, actions and proposed etc for all inputs where the ball starts in the middle of the screen.
  # But i need to pick up the sequences which are exactly like one another.  
  x = np.sum(seqInputs,axis=1)[0,:,:] #3:17
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
  dseqOutputs4comp = {pop:dseqOutputs[pop][goodInds,:,:,:] for pop in dseqOutputs.keys()}
  summedInputSequences = np.sum(seqInputs4comp,axis=1)
  dsummedOutputs = {pop:np.zeros((len(goodInds),nbseq),dtype=float) for pop in lmotorpop}
  for pop in lmotorpop:
    dsummedOutputs[pop] = np.sum(np.sum(dseqOutputs4comp[pop],axis=2),axis=2)
  lSeqNBs4comp = [0,1,2,3,4,5,6,7,8,9,10]
  fig, axs = plt.subplots(6, 5, figsize=(10,8));
  lax = axs.ravel()
  for i in range(5):
    cSeq = lSeqNBs4comp[i]
    if i<len(goodInds):
      lax[i].imshow(summedInputSequences[cSeq,:,:])
      lax[i].axis('off')
      for pop,clr in zip(lmotorpop,['b','r','g']):
        lax[i+5].plot(dsummedOutputs[pop][cSeq,:],clr+'-o',markersize=3)
      if i==0: lax[i+5].set_ylabel('# of pop spikes')
      lax[i+10].plot(seqActions4comp[cSeq,:],'-o',color=(0,0,0,1),markersize=3)
      lax[i+10].plot(seqPropActions[cSeq,:],'-o',color=(0.5,0.5,0.5,1),markersize=3)
      lax[i+10].set_yticks([1,3,4])
      if i==0: lax[i+10].set_yticklabels(['STAY','DOWN','UP'])
    cSeq = lSeqNBs4comp[i+5]
    if (i+5)<len(goodInds):
      lax[i+15].imshow(summedInputSequences[cSeq,:,:])
      lax[i+15].axis('off')
      for pop,clr in zip(lmotorpop,['b','r','g']):
        lax[i+20].plot(dsummedOutputs[pop][cSeq,:],clr+'-o',markersize=3)
      if i==0: lax[i+20].set_ylabel('# of pop spikes')
      lax[i+25].plot(seqActions4comp[cSeq,:],'-o',color=(0,0,0,1),markersize=3)
      lax[i+25].plot(seqPropActions[cSeq,:],'-o',color=(0.5,0.5,0.5,1),markersize=3)
      lax[i+25].set_yticks([1,3,4])
      if i==0: lax[i+25].set_yticklabels(['STAY','DOWN','UP'])
    if i==0:
      lax[i+5].legend(lmotorpop,loc='best')
      lax[i+10].legend(['Actions','Proposed'],loc='best')

def analyzeActionLearningForRepeatedInputSequences(dact, InputImages, BallPixel=(10,10), RacketPixel=(5,17)):
  nbseq=2
  targetCorr=0.99
  ballInds = np.where(InputImages[:,BallPixel[0],BallPixel[1]]>250)
  racketInds = np.where(InputImages[:,RacketPixel[0],RacketPixel[1]]>250)
  targetInds = []
  for inds in racketInds[0]:
    if inds in ballInds[0]:
      targetInds.append(inds)
  seqImages = []
  for inds in targetInds:
    seqImages.append(np.sum(InputImages[inds:inds+1,:,3:17],0))
  x = seqImages[0]
  goodInds = []
  for j in range(np.shape(seqImages)[0]):
    y = seqImages[j]
    corr, p_value = pearsonr(x.flat, y.flat)
    if corr>targetCorr:
      goodInds.append(j)
  lmotorpop = [pop for pop in dconf['net']['EMotorPops'] if dconf['net']['allpops'][pop]>0] 
  goodSeqImages = []
  repActions = []
  repPropActions = []
  repRewards = []
  dseqOutputs = {pop:np.zeros((len(goodInds),dact[pop].shape[1],dact[pop].shape[2]),dtype=float) for pop in lmotorpop}
  for inds in goodInds:
    goodSeqImages.append(np.sum(InputImages[targetInds[inds]:targetInds[inds]+2,:,:],0))
    repActions.append(actreward['action'][targetInds[inds]+1])
    repRewards.append(actreward['reward'][targetInds[inds]+1])
    repPropActions.append(actreward['proposed'][targetInds[inds]+1])
    for pop in dseqOutputs.keys():
      dseqOutputs[pop][inds,:,:]=dact[pop][targetInds[inds]+1,:,:]
  dsummedOutputs = {pop:np.zeros((len(goodInds),1),dtype=float) for pop in lmotorpop}
  for pop in lmotorpop:
    dsummedOutputs[pop] = np.sum(np.sum(dseqOutputs[pop],axis=2),axis=1)
  fig, axs = plt.subplots(3, 1, figsize=(10,8));
  lax = axs.ravel()
  lax[0].imshow(np.sum(goodSeqImages,0))
  lax[0].axis('off')
  for pop,clr in zip(lmotorpop,['b','r','g']):
    lax[1].plot(dsummedOutputs[pop],clr+'-o',markersize=3)
    lax[1].set_ylabel('# of pop spikes')
  lax[1].legend(lmotorpop,loc='best')
  lax[2].plot(repActions,'-o',color=(0,0,0,1),markersize=3)
  lax[2].plot(repPropActions,'-o',color=(0.5,0.5,0.5,1),markersize=3)
  lax[2].set_yticks([1,3,4])
  lax[2].set_yticklabels(['STAY','DOWN','UP'])
  lax[2].legend(['Actions','Proposed'],loc='best')

def analyzeRepeatedInputForSingleEvent(dact, InputImages, targetPixel=(10,10)):
  midInds = np.where(InputImages[:,targetPixel[0],targetPixel[1]]>250)
  repSeqInds = []
  for i in range(len(midInds[0])-1):
    if midInds[0][i+1]-midInds[0][i]<5:
      repSeqInds.append(i+1)
  uniqueSeqStartInds = []
  for i in range(len(midInds[0])):
    if i not in repSeqInds:
      uniqueSeqStartInds.append(midInds[0][i])
  # for each midInd, find 14 (13 could be enough but i am not sure) consecutive Images to see the trajectory.
  ImgH, ImgW = InputImages.shape[1],InputImages.shape[2]
  lmotorpop = [pop for pop in dconf['net']['EMotorPops'] if dconf['net']['allpops'][pop]>0] 
  seqInputs = np.zeros((len(uniqueSeqStartInds),ImgH,ImgW),dtype=float)
  seqActions = np.zeros((len(uniqueSeqStartInds),1),dtype=float)
  seqPropActions = np.zeros((len(uniqueSeqStartInds),1),dtype=float)
  seqRewards = np.zeros((len(uniqueSeqStartInds),1),dtype=float)
  seqHitMiss = np.zeros((len(uniqueSeqStartInds),1),dtype=float)
  dseqOutputs = {pop:np.zeros((len(uniqueSeqStartInds),dact[pop].shape[1],dact[pop].shape[2]),dtype=float) for pop in lmotorpop}
  for i in range(len(uniqueSeqStartInds)):
    cInputInd = uniqueSeqStartInds[i]
    seqInputs[i,:,:] = InputImages[cInputInd,:,:]
    seqActions[i] = actreward['action'][cInputInd]
    seqRewards[i] = actreward['reward'][cInputInd]
    seqPropActions[i] = actreward['proposed'][cInputInd]
    seqHitMiss[i] = actreward['hit'][cInputInd]
    for pop in dseqOutputs.keys():
        dseqOutputs[pop][i,:,:]=dact[pop][cInputInd,:,:]
  dFR = {pop:np.sum(np.sum(dseqOutputs[pop],axis=1),axis=1) for pop in lmotorpop}
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
  pop_ind = 1
  for pop in lmotorpop:
    lax[pop_ind+2].hist(dFR[pop])
    lax[pop_ind+2].set_xlabel('# of EMUP spikes')
    pop_ind = pop_ind+1
  fig, axs = plt.subplots(4, 1, figsize=(10,8));
  lax = axs.ravel()
  lax[0].imshow(np.sum(seqInputs,axis=0))
  lax[0].axis('off')
  for pop,clr in zip(lmotorpop,['b','r','g']):
    lax[1].plot(np.sum(np.sum(dseqOutputs[pop],axis=1),axis=1),clr+'-o',markersize=3)
  lax[1].set_ylabel('# of pop spikes')
  lax[2].plot(seqActions,'-o',color=(0,0,0,1),markersize=3)
  lax[2].plot(seqPropActions,'-o',color=(0.5,0.5,0.5,1),markersize=3)
  lax[2].set_yticks([1,3,4])
  lax[2].set_yticklabels(['STAY','DOWN','UP'])
  lax[3].plot(seqRewards ,'-o',color=(0,0,0,1),markersize=3)
  lax[3].plot(seqHitMiss,'-o',color=(0.5,0.5,0.5,1),markersize=3)
  lax[3].set_yticks([-1,0,1])
  lax[3].legend(['Rewards','Hit/Moss'])
  lax[1].legend(lmotorpop,loc='best')
  lax[2].legend(['Actions','Proposed'],loc='best')

def plotAllWeightsChangePreMtoM(pdf, dstartidx, dendidx, targetpop ,tpnt1 = 0, tpnt2 = -1):
  utimes = np.unique(pdf.time)
  nbNeurons = dendidx[targetpop]+1-dstartidx[targetpop]
  tpnts = len(utimes)
  wts_top = np.zeros((tpnts,nbNeurons))
  count = 0
  for idx in range(dstartidx[targetpop],dendidx[targetpop]+1,1): # first plot average weight onto each individual neuron
    pdfs = pdf[(pdf.postid==idx)]  
    wts = [np.mean(pdfs[(pdfs.time==t)].weight) for t in utimes]
    wts_top[:,count] = wts
    count = count+1
  dim_neurons = int(np.sqrt(nbNeurons))
  avgwt_tpnt1 = np.reshape(wts_top[tpnt1,:],(dim_neurons,dim_neurons))
  avgwt_tpnt2 = np.reshape(wts_top[tpnt2,:],(dim_neurons,dim_neurons))
  plt.imshow(np.subtract(avgwt_tpnt2,avgwt_tpnt1))
  plt.title('Change in weights-->'+targetpop)
  plt.colorbar()

def plotWeightsChangeOnePreMtoM(pdf, dstartidx, dendidx, prepop , targetpop ,tpnt1 = 0, tpnt2 = -1,drawplot=False):
  utimes = np.unique(pdf.time)
  nbNeurons = dendidx[targetpop]+1-dstartidx[targetpop]
  tpnts = len(utimes)
  wts_top = np.zeros((tpnts,nbNeurons))
  count = 0
  prestartidx = dstartidx[prepop]
  preendidx = dendidx[prepop]
  for idx in range(dstartidx[targetpop],dendidx[targetpop]+1,1): # first plot average weight onto each individual neuron
    pdfs = pdf[(pdf.postid==idx) & (pdf.preid>=prestartidx) & (pdf.preid<=preendidx)]  
    wts = [np.mean(pdfs[(pdfs.time==t)].weight) for t in utimes]
    wts_top[:,count] = wts
    count = count+1
  dim_neurons = int(np.sqrt(nbNeurons))
  avgwt_tpnt1 = np.reshape(wts_top[tpnt1,:],(dim_neurons,dim_neurons))
  avgwt_tpnt2 = np.reshape(wts_top[tpnt2,:],(dim_neurons,dim_neurons))
  if drawplot:
    plt.imshow(np.subtract(avgwt_tpnt2,avgwt_tpnt1))
    plt.title('Change in weights '+prepop+' to '+targetpop)
    plt.colorbar()
  return np.subtract(avgwt_tpnt2,avgwt_tpnt1)

def plotWeightChangeOnePreMtoMAll(pdf, dstartidx, dendidx, tpnt1 = 0, tpnt2 = -1, figsize=(14,8)):
  minV = 0
  maxV = 0
  weightChanges = dict()
  for prepop in dconf['net']['EPreMPops']:
    if dconf['net']['allpops'][prepop]>0:
      for targetpop in dconf['net']['EMotorPops']:
        if dconf ['net']['allpops'][targetpop]>0:
          weightChanges[prepop+'->'+targetpop] = plotWeightsChangeOnePreMtoM(pdf, dstartidx, dendidx, prepop = prepop , targetpop=targetpop)
          if np.amin(weightChanges[prepop+'->'+targetpop])<minV: minV = np.amin(weightChanges[prepop+'->'+targetpop])
          if np.amax(weightChanges[prepop+'->'+targetpop])>maxV: maxV = np.amax(weightChanges[prepop+'->'+targetpop])
  nbrows = 4
  nbcols = int(np.ceil(len(weightChanges)/4))
  fig, axs = plt.subplots(nbrows, nbcols, figsize=figsize);
  lax = axs.ravel()
  cbaxes = fig.add_axes([0.92, 0.4, 0.01, 0.2])
  conn_count = 0
  for conns in weightChanges.keys():
    pcm = lax[conn_count].imshow(weightChanges[conns],vmin = minV, vmax = maxV)
    lax[conn_count].set_ylabel(conns,fontsize=8)
    conn_count = conn_count+1
    if conn_count==len(weightChanges): plt.colorbar(pcm, cax = cbaxes)
  for _ in range(conn_count,nbrows*nbcols):
    lax[conn_count].set_axis_off()
    conn_count = conn_count+1


def plotConns(prepop,postpop):
  fn = 'data/'+dconf['sim']['name']+'synConns.pkl'
  D = pickle.load(open(fn,'rb'))
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for conns in D.keys():
    if conns==prepop+'->'+postpop:
      cConns = D[conns]['blist']
      cCoords = D[conns]['coords']
  for i in range(np.shape(cCoords)[0]):
    prex, prey, postx, posty = cCoords[i][0],cCoords[i][1],cCoords[i][2],cCoords[i][3]
    ax.plot([prex,postx],[prey,posty],[9,0],'ro-')
    ax.set_zticks([0,9])
    ax.set_zticklabels([postpop,prepop])
  plt.show()

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

def gifpath (): return 'gif/' + getdatestr() + dconf['sim']['name']

if __name__ == '__main__':
  stepNB = -1
  if len(sys.argv) > 1:
    try:
      stepNB = int(sys.argv[1]) #which file(stepNB) want to plot
    except:
      pass
  print(stepNB)
  allpossible_pops = ['ER','IR','EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE','IV1','EV4','IV4','EMT','IMT','EMDOWN','EMUP','IM']
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
  #fig=animInput(InputImages,gifpath()+'_input.mp4')  
  #figure(); drawcellVm(simConfig,lclr=['r','g','b','c','m','y'])
  if totalDur <= 10e3:
    pravgrates(dspkT,dspkID,dnumc,tlim=(totalDur-1e3,totalDur))
    drawraster(dspkT,dspkID)
    figure(); drawcellVm(simConfig,lclr=['r','g','b','c','m','y'])    
  else:
    pravgrates(dspkT,dspkID,dnumc,tlim=(250,totalDur))    

