import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from conf import dconf
from collections import OrderedDict
from pylab import *
import os
import anim
from matplotlib import animation
from simdat import loadInputImages, loadsimdat, loadMotionFields, loadObjPos
from imgutils import getoptflow, getoptflowframes

rcParams['font.size'] = 12

InputImages = loadInputImages(dconf['sim']['name'])
simConfig, pdf, actreward, dstartidx, dendidx, dnumc, dspkID, dspkT = loadsimdat(dconf['sim']['name'])
ldflow = loadMotionFields(dconf['sim']['name'])

totalDur = int(dconf['sim']['duration'])
tstepPerAction = dconf['sim']['tstepPerAction'] # time step per action (in ms)

lpop = ['ER', 'EV1', 'EV4', 'EMT', 'IR', 'IV1', 'IV4', 'IMT',\
        'EV1DW','EV1DNW', 'EV1DN', 'EV1DNE','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE',\
        'EMDOWN','EMUP']

ddir = OrderedDict({'EV1DW':'W','EV1DNW':'NW', 'EV1DN':'N','EV1DNE':'NE','EV1DE':'E','EV1DSW':'SW','EV1DS':'S','EV1DSE':'SE'})

t1 = range(0,totalDur,tstepPerAction)
t2 = range(tstepPerAction,totalDur+tstepPerAction,tstepPerAction)

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

dact = {pop:generateActivityMap(t1, t2, dspkT[pop], dspkID[pop], dnumc[pop], dstartidx[pop]) for pop in lpop}
dmaxSpk = OrderedDict({pop:np.max(dact[pop]) for pop in lpop})
max_spks = np.max([dmaxSpk[p] for p in lpop])

#
def plotActivityMaps (pauset=1, gifpath=None, mp4path=None, framerate=5, zf=10):
  # plot activity in different layers as a function of input images
  #fig, axs = plt.subplots(4, 5, figsize=(12,6)); lax = axs.ravel()
  fig, axs = plt.subplots(4, 5); lax = axs.ravel()
  cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2]) 
  ltitle = ['Input Images', 'Excit R', 'Excit V1', 'Excit V4', 'Excit MT', 'Inhib R', 'Inhib V1', 'Inhib V4', 'Inhib MT']
  for p in ddir.keys(): ltitle.append(ddir[p])
  lact = [InputImages]; lvmax = [255]; xl = [(-.5,19.5)]; yl = [(19.5,-0.5)]
  lfnimage = []
  for pop in lpop:
    lact.append(dact[pop])
    lvmax.append(max_spks)
    xl.append( (-0.5, lact[-1].shape[1] - 0.5) )
    yl.append( (lact[-1].shape[1] - 0.5, -0.5))
  for t in range(1,len(t1)):
    fig.suptitle('Time = ' + str(t*tstepPerAction) + ' ms')
    idx = 0
    for ldx,ax in enumerate(lax):
      if ldx == 5 or idx > len(dact.keys()):
        ax.axis('off')
        continue
      if ldx==0: offidx=-1
      else: offidx=0
      pcm = ax.imshow( lact[idx][t+offidx,:,:], origin='upper', cmap='gray', vmin=0, vmax=lvmax[idx])
      ax.set_xlim(xl[idx]) 
      ax.set_ylim(yl[idx])
      ax.set_ylabel(ltitle[idx])
      if ldx==2: plt.colorbar(pcm, cax = cbaxes)  
      idx += 1
    if gifpath is not None or mp4path is not None:
      fnimg = '/tmp/'+str(t).zfill(zf)+'.png'
      savefig(fnimg); lfnimage.append(fnimg)
    if pauset > 0: plt.pause(pauset)
  if gifpath is not None: anim.savegif(lfnimage, gifpath)
  if mp4path is not None: anim.savemp4('/tmp/*.png', mp4path, framerate)
  for fn in lfnimage: os.unlink(fn) # remove the tmp files
  return fig, axs, plt

#
def animActivityMaps (outpath='gif/'+dconf['sim']['name']+'actmap.mp4', framerate=10, figsize=(18,10), dobjpos=None):
  # plot activity in different layers as a function of input images  
  ioff()
  if figsize is not None: fig, axs = plt.subplots(4, 5, figsize=figsize);
  else: fig, axs = plt.subplots(4, 5);
  lax = axs.ravel()
  cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2]) 
  ltitle = ['Input Images', 'Excit R', 'Excit V1', 'Excit V4', 'Excit MT', 'Inhib R', 'Inhib V1', 'Inhib V4', 'Inhib MT']
  for p in ddir.keys(): ltitle.append(ddir[p])
  for p in ['Excit M DOWN', 'Excit M UP']: ltitle.append(p)
  lact = [InputImages]; lvmax = [255];
  lfnimage = []
  for pop in lpop:
    lact.append(dact[pop])
    lvmax.append(max_spks)
  ddat = {}
  fig.suptitle('Time = ' + str(0*tstepPerAction) + ' ms')
  idx = 0
  objfctr = 1.0
  if dconf['DirectionDetectionAlgo']['UseFull']: objfctr=1/8.  
  for ldx,ax in enumerate(lax):
    if idx > len(dact.keys()):
      ax.axis('off')
      continue
    if ldx==0:
      offidx=-1
    elif ldx==5:
      offidx=1
    else:
      offidx=0
    if ldx==5:
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
      if ldx==0 or ldx==5:
        offidx=-1
      else:
        offidx=0
      if ldx == 5:
        ddat[ldx].set_UVC(ldflow[t+offidx]['thflow'][:,:,0],-ldflow[t]['thflow'][:,:,1])        
      else:
        ddat[ldx].set_data(lact[idx][t+offidx,:,:])
        if ldx==0 and dobjpos is not None:
          lobjx,lobjy = [objfctr*dobjpos[k][t,0] for k in dobjpos.keys()], [objfctr*dobjpos[k][t,1] for k in dobjpos.keys()]
          ddat['objpos'].set_data(lobjx,lobjy)        
        idx += 1
    return fig
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=len(t1)-1)
  writer = anim.getwriter(outpath, framerate=framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  ion()
  return fig, axs, plt

#
def animInput (InputImages, outpath, framerate=10, figsize=None, showflow=True, ldflow=None, dobjpos=None):
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
  nframe = len(t1)
  if showflow: nframe-=1
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=nframe)
  writer = anim.getwriter(outpath, framerate=framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  ion()
  return fig

#
def getmaxdir (dact, ddir):
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
  if figsize is not None: fig, axs = plt.subplots(1, 3, figsize=figsize);
  else: fig, axs = plt.subplots(1, 3);
  lax = axs.ravel()
  ltitle = ['Input Images', 'Motion', 'Detected Motion']
  lact = [InputImages]; lvmax = [255];
  lfnimage = []
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


#fig, axs, plt = animActivityMaps('test2.mp4')
# fig, axs, plt = animActivityMaps('gif/'+dconf['sim']['name']+'actmap.mp4', framerate=10)

