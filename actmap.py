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
from simdat import loadInputImages, loadsimdat, loadMotionFields
from imgutils import getoptflow

rcParams['font.size'] = 6

InputImages = loadInputImages(dconf['sim']['name'])
simConfig, pdf, actreward, dstartidx, dendidx, dnumc = loadsimdat(dconf['sim']['name'])
ldflow = loadMotionFields(dconf['sim']['name'])

totalDur = int(dconf['sim']['duration'])
tstepPerAction = dconf['sim']['tstepPerAction'] # time step per action (in ms)

spkID= np.array(simConfig['simData']['spkid'])
spkT = np.array(simConfig['simData']['spkt'])

lpop = ['ER', 'EV1', 'EV4', 'EMT', 'IR', 'IV1', 'IV4', 'IMT',\
        'EV1DW','EV1DNW', 'EV1DN', 'EV1DNE','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE']

ddir = OrderedDict({'EV1DW':'W','EV1DNW':'NW', 'EV1DN':'N','EV1DNE':'NE','EV1DE':'E','EV1DSW':'SW','EV1DS':'S','EV1DSE':'SE'})

dspkID,dspkT = {},{}
for pop in lpop:
  dspkID[pop] = spkID[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
  dspkT[pop] = spkT[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]

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
def animActivityMaps (outpath, framerate=10, figsize=(7,3)):
  ioff()
  # plot activity in different layers as a function of input images
  if figsize is not None: fig, axs = plt.subplots(4, 5, figsize=figsize);
  else: fig, axs = plt.subplots(4, 5);
  lax = axs.ravel()
  cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2]) 
  ltitle = ['Input Images', 'Excit R', 'Excit V1', 'Excit V4', 'Excit MT', 'Inhib R', 'Inhib V1', 'Inhib V4', 'Inhib MT']
  for p in ddir.keys(): ltitle.append(ddir[p])
  lact = [InputImages]; lvmax = [255];
  lfnimage = []
  for pop in lpop:
    lact.append(dact[pop])
    lvmax.append(max_spks)
  ddat = {}
  fig.suptitle('Time = ' + str(0*tstepPerAction) + ' ms')
  idx = 0
  for ldx,ax in enumerate(lax):
    if ldx == 5 or idx > len(dact.keys()):
      ax.axis('off')
      if ldx == 5:
        X, Y = np.meshgrid(np.arange(0, InputImages[0].shape[1], 1), np.arange(0,InputImages[0].shape[0],1))
        ddat[ldx] = ax.quiver(X,Y,ldflow[0]['flow'][:,:,0],-ldflow[0]['flow'][:,:,1], pivot='mid', units='inches',width=0.022,scale=1/0.15)
        ax.set_xlim((0,InputImages[0].shape[1])); ax.set_ylim((0,InputImages[0].shape[0]))
        ax.invert_yaxis()        
        pass
      continue
    if ldx==0: offidx=-1
    else: offidx=0
    pcm = ax.imshow(lact[idx][offidx,:,:],origin='upper',cmap='gray',vmin=0,vmax=lvmax[idx])
    ddat[ldx] = pcm
    ax.set_ylabel(ltitle[idx])
    if ldx==2: plt.colorbar(pcm, cax = cbaxes)  
    idx += 1
  def updatefig (t):
    print('frame t = ', str(t*tstepPerAction))
    fig.suptitle('Time = ' + str(t*tstepPerAction) + ' ms')
    idx = 0
    for ldx,ax in enumerate(lax):
      if ldx == 5 or idx > len(dact.keys()):
        if ldx == 5:
          
          pass
        continue
      if ldx==0: offidx=-1
      else: offidx=0
      ddat[ldx].set_data(lact[idx][t+offidx,:,:])
      idx += 1
    return fig
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=len(t1))
  writer = anim.getwriter(outpath, framerate=framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  ion()
  return fig, axs, plt

#
def animInput (InputImages, outpath, framerate=10, figsize=None, showflow=True, ldflow=None):
  # animate the input images; showflow specifies whether to calculate/animate optical flow
  ioff()
  # plot input images and optionally optical flow
  ncol = 1
  if showflow: ncol+=1
  if figsize is not None:
    fig = figure(figsize=figsize)
  else:
    fig = figure()
  lax = [subplot(1,ncol,i+1) for i in range(ncol)]
  ltitle = ['Input Images']
  lact = [InputImages]; lvmax = [255]; xl = [(-.5,19.5)]; yl = [(19.5,-0.5)]
  ddat = {}
  fig.suptitle('Time = ' + str(0*tstepPerAction) + ' ms')
  idx = 0
  lflow = []
  if showflow and ldflow is None: ldflow = getoptflowframes(InputImages)
  for ldx,ax in enumerate(lax):
    if ldx==0:
      pcm = ax.imshow( lact[idx][0,:,:], origin='upper', cmap='gray', vmin=0, vmax=lvmax[idx])
      ddat[ldx] = pcm
      ax.set_ylabel(ltitle[idx])
    else:
      X, Y = np.meshgrid(np.arange(0, InputImages[0].shape[1], 1), np.arange(0,InputImages[0].shape[0],1))
      ddat[ldx] = ax.quiver(X,Y,ldflow[0]['flow'][:,:,0],-ldflow[0]['flow'][:,:,1], pivot='mid', units='inches',width=0.022,scale=1/0.15)
      ax.set_xlim((0,InputImages[0].shape[1])); ax.set_ylim((0,InputImages[0].shape[0]))
      ax.invert_yaxis()
    idx += 1
  def updatefig (t):
    print('frame t = ', str(t*tstepPerAction))
    fig.suptitle('Time = ' + str(t*tstepPerAction) + ' ms')
    for ldx,ax in enumerate(lax):
      if ldx == 0:
        ddat[ldx].set_data(lact[0][t-1,:,:])
      else:
        ddat[ldx].set_UVC(ldflow[t]['flow'][:,:,0],-ldflow[t]['flow'][:,:,1])        
    return fig
  nframe = len(t1)
  if showflow: nframe-=1
  ani = animation.FuncAnimation(fig, updatefig, interval=1, frames=nframe)
  writer = anim.getwriter(outpath, framerate=framerate)
  ani.save(outpath, writer=writer); print('saved animation to', outpath)
  ion()
  return fig

#fig, axs, plt = animActivityMaps('test2.mp4')
# fig, axs, plt = animActivityMaps('data/'+dconf['sim']['name']+'actmap.mp4', framerate=10)

