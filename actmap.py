import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from conf import dconf
from collections import OrderedDict
from pylab import *
import os
import anim

Input_Images = np.loadtxt('data/'+dconf['sim']['name']+'InputImages.txt')
New_InputImages = []
NB_Images = int(Input_Images.shape[0]/Input_Images.shape[1])
for x in range(NB_Images):
    fp = x*Input_Images.shape[1]
    cImage = Input_Images[fp:fp+20,:] # 20 is sqrt of 400 (20x20 pixels)
    New_InputImages.append(cImage)
New_InputImages = np.array(New_InputImages)

##Change the rasterdata file below
phl_file = open('data/'+dconf['sim']['name']+'RasterData.pkl','rb')
data1 = pickle.load(phl_file)
spkTimes = data1["spkTimes"]
spkInds = data1["spkInds"]
cellIDs = data1["cellGids"]
skColors = data1["spkColors"] 

totalDur = int(dconf['sim']['duration'])
tBin_Size = 100

simConfig = pickle.load(open('data/'+dconf['sim']['name']+'simConfig.pkl','rb'))
dstartidx = {p:simConfig['net']['pops'][p]['cellGids'][0] for p in simConfig['net']['pops'].keys()} # starting indices for each population
dendidx = {p:simConfig['net']['pops'][p]['cellGids'][-1] for p in simConfig['net']['pops'].keys()} # ending indices for each population
dnumc = {p:dendidx[p]-dstartidx[p]+1 for p in simConfig['net']['pops'].keys()}

spkID= np.array(spkInds)
spkT = np.array(spkTimes)

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
def plotActivityMaps (pauset=1, gifpath=None, mp4path=None, framerate=5):
  # plot activity in different layers as a function of input images
  fig, axs = plt.subplots(4, 5, figsize=(12,6)); lax = axs.ravel()
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
    fig.suptitle(str(tBin_Size)+' ms binned activity ' + str(t*tBin_Size) + ' ms')
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
      fnimg = '/tmp/'+str(t)+'.png'
      savefig(fnimg); lfnimage.append(fnimg)
    if pauset > 0: plt.pause(pauset)
  if gifpath is not None: anim.savegif(lfnimage, gifpath)
  if mp4path is not None: anim.savemp4('/tmp/*.png', mp4path, framerate)
  for fn in lfnimage: os.unlink(fn) # remove the tmp files
  return fig, axs, plt

fig, axs, plt = plotActivityMaps(pauset=0,mp4path='data/'+dconf['sim']['name']+'actmap.mp4', framerate=10)

  
