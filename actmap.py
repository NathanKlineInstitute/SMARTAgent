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
from simdat import loadInputImages, loadsimdat

rcParams['font.size'] = 6

New_InputImages = loadInputImages('data/'+dconf['sim']['name']+'InputImages.txt')
simConfig, pdf, actreward, dstartidx, dendidx, dnumc = loadsimdat(dconf['sim']['name'])

totalDur = int(dconf['sim']['duration'])
tBin_Size = 100

spkID= np.array(simConfig['simData']['spkid'])
spkT = np.array(simConfig['simData']['spkt'])

lpop = ['ER', 'EV1', 'EV4', 'EMT', 'IR', 'IV1', 'IV4', 'IMT',\
        'EV1DW','EV1DNW', 'EV1DN', 'EV1DNE','EV1DE','EV1DSW', 'EV1DS', 'EV1DSE']

ddir = OrderedDict({'EV1DW':'W','EV1DNW':'NW', 'EV1DN':'N','EV1DNE':'NE','EV1DE':'E','EV1DSW':'SW','EV1DS':'S','EV1DSE':'SE'})

dspkID,dspkT = {},{}
for pop in lpop:
  dspkID[pop] = spkID[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
  dspkT[pop] = spkT[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]

t1 = range(0,totalDur,tBin_Size)
t2 = range(tBin_Size,totalDur+tBin_Size,tBin_Size)

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
  lact = [New_InputImages]; lvmax = [255]; xlim = [(-.5,19.5)]; ylim = [(19.5,-0.5)]
  lfnimage = []
  for pop in lpop:
    lact.append(dact[pop])
    lvmax.append(max_spks)
    xlim.append( (-0.5, lact[-1].shape[1] - 0.5) )
    ylim.append( (lact[-1].shape[1] - 0.5, -0.5))
  for t in range(1,len(t1)):
    fig.suptitle('Time = ' + str(t*tBin_Size) + ' ms')
    idx = 0
    for ldx,ax in enumerate(lax):
      if ldx == 5 or idx > len(dact.keys()):
        ax.axis('off')
        continue
      if ldx==0: offidx=-1
      else: offidx=0
      pcm = ax.imshow( lact[idx][t+offidx,:,:], origin='upper', cmap='gray', vmin=0, vmax=lvmax[idx])
      ax.set_xlim(xlim[idx]) 
      ax.set_ylim(ylim[idx])
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
  if figsize is not None:
    fig, axs = plt.subplots(4, 5, figsize=figsize);
  else:
    fig, axs = plt.subplots(4, 5);
  lax = axs.ravel()
  cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2]) 
  ltitle = ['Input Images', 'Excit R', 'Excit V1', 'Excit V4', 'Excit MT', 'Inhib R', 'Inhib V1', 'Inhib V4', 'Inhib MT']
  for p in ddir.keys(): ltitle.append(ddir[p])
  lact = [New_InputImages]; lvmax = [255]; xlim = [(-.5,19.5)]; ylim = [(19.5,-0.5)]
  lfnimage = []
  for pop in lpop:
    lact.append(dact[pop])
    lvmax.append(max_spks)
    xlim.append( (-0.5, lact[-1].shape[1] - 0.5) )
    ylim.append( (lact[-1].shape[1] - 0.5, -0.5))
  ddat = {}
  fig.suptitle('Time = ' + str(0*tBin_Size) + ' ms')
  idx = 0
  for ldx,ax in enumerate(lax):
    if ldx == 5 or idx > len(dact.keys()):
      ax.axis('off')
      continue
    if ldx==0: offidx=-1
    else: offidx=0
    pcm = ax.imshow( lact[idx][offidx,:,:], origin='upper', cmap='gray', vmin=0, vmax=lvmax[idx])
    ddat[ldx] = pcm
    ax.set_xlim(xlim[idx]) 
    ax.set_ylim(ylim[idx])
    ax.set_ylabel(ltitle[idx])
    if ldx==2: plt.colorbar(pcm, cax = cbaxes)  
    idx += 1
  def updatefig (t):
    print('frame t = ', str(t*tBin_Size))
    fig.suptitle('Time = ' + str(t*tBin_Size) + ' ms')
    idx = 0
    for ldx,ax in enumerate(lax):
      if ldx == 5 or idx > len(dact.keys()): continue
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

fig, axs, plt = animActivityMaps('data/'+dconf['sim']['name']+'actmap.mp4', framerate=10)

