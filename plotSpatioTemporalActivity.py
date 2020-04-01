import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from conf import dconf
from collections import OrderedDict
from pylab import *

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

neuronIDs = np.unique(spkInds)

AllCells = np.array(spkInds)
AllCells_spkTimes = np.array(spkTimes)

lpop = ['ER', 'EV1', 'EV4', 'EIT', 'IR', 'IV1', 'IV4', 'IIT']

dspkID,dspkT = {},{}
for pop in lpop:
  dspkID[pop] = AllCells[(AllCells >= dstartidx[pop]) & (AllCells <= dendidx[pop])]
  dspkT[pop] = AllCells_spkTimes[(AllCells >= dstartidx[pop]) & (AllCells <= dendidx[pop])]

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
def plotActivityMaps (pauset=1):
  fig, axs = plt.subplots(2, 5,figsize=(12,6)); lax = axs.ravel()
  cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2]) 
  ltitle = ['Input Images', 'Excit R', 'Excit V1', 'Excit V4', 'Excit IT', 'Inhib R', 'Inhib V1', 'Inhib V4', 'Inhib IT']
  lact = [New_InputImages]; lvmax = [255]; llim = [(-.5,19.5)]
  for pop in lpop:
    lact.append(dact[pop])
    lvmax.append(max_spks)
    llim.append( (-0.5, lact[-1].shape[1] - 0.5) )
  for t in range(1,len(t1)):
    fig.suptitle(str(tBin_Size)+' ms binned activity ' + str(t*tBin_Size) + ' ms')
    idx = 0
    for ldx,ax in enumerate(lax):
      if ldx == 5:
        ax.axis('off')
        continue
      if ldx==0: offidx=-1
      else: offidx=0
      pcm = ax.imshow( lact[idx][t+offidx,:,:], cmap='gray', vmin=0, vmax=lvmax[idx])
      ax.set_xlim(llim[idx]); ax.set_ylim(llim[idx]); ax.set_title(ltitle[idx])
      if ldx==2: plt.colorbar(pcm, cax = cbaxes)  
      idx += 1
    plt.pause(pauset)  
  return fig, axs, plt

fig, axs, plt = plotActivityMaps()
