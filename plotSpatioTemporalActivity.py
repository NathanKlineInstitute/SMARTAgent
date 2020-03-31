import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from conf import dconf
from collections import OrderedDict

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

def generateActivityMap(t1, t2, spkt, spkID, numc):
  Nact = np.zeros(shape=(int(np.sqrt(N)),int(np.sqrt(N)),len(t1)))
  """
  for t,ID in zip(spkt,spkID):
    Nact[i][j][t]
    N = NCells
  """    
  for i in range(int(np.sqrt(N))):
    for j in range(int(np.sqrt(N))):
      cNeuronID = j+(i*int(np.sqrt(N)))
      cNeuron_spkt = NCells_spkTimes[NCells==cNeuronID]
      for t in range(len(t1)):
        cbinSpikes = cNeuron_spkt[(cNeuron_spkt>t1[t]) & (cNeuron_spkt<=t2[t])]
        Nact[i][j][t] = len(cbinSpikes)
  return Nact

dact = {pop:generateActivityMap(t1, t2, dnumc[pop], dspkT[pop]) for pop in lpop}

dmaxSpk = OrderedDict({pop:np.max(dact[pop]) for pop in lpop})
max_spks = np.max([dmaxSpk[p] for p in lpop])

fig, axs = plt.subplots(2, 5,figsize=(12,6))
ax = axs.ravel()

cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2]) 
#cb = plt.colorbar(ax1, cax = cbaxes)  

for t in range(1,len(t1)):
    fig.suptitle(str(tBin_Size)+' ms binned activity ' + str(t*tBin_Size) + ' ms')
    ax[0].imshow(New_InputImages[t-1,:,:],cmap='gray', vmin=0, vmax = 255)
    ax[0].set_xlim(-0.5,19.5)
    ax[0].set_ylim(-0.5,19.5)
    ax[0].set_title('Input Images')
    ax[1].imshow(dact['ER'][:,:,t],cmap='gray', vmin=0, vmax=max_spks)
    ax[1].set_title('Excit R')
    ax[1].set_xlim(-0.5,19.5)
    ax[1].set_ylim(-0.5,19.5)
    pcm2 = ax[2].imshow(dact['EV1'][:,:,t],cmap='gray', vmin=0, vmax=max_spks)
    ax[2].set_xlim(-0.5,19.5)
    ax[2].set_ylim(-0.5,19.5)
    ax[2].set_title('Excit V1')
    ax[3].imshow(dact['EV4'][:,:,t],cmap='gray', vmin=0, vmax=max_spks)
    ax[3].set_xlim(-0.5,9.5)
    ax[3].set_ylim(-0.5,9.5)
    ax[3].set_title('Excit V4')
    ax[4].imshow(dact['EIT'][:,:,t],cmap='gray', vmin=0, vmax=max_spks)
    ax[4].set_xlim(-0.5,4.5)
    ax[4].set_ylim(-0.5,4.5)
    ax[4].set_title('Excit IT')
    ax[6].imshow(dact['IR'][:,:,t],cmap='gray', vmin=0, vmax=max_spks)
    ax[6].set_xlim(-0.5,9.5)
    ax[6].set_ylim(-0.5,9.5)
    ax[6].set_title('Inhib R')
    ax[5].axis('off')
    ax[7].imshow(dact['IV1'][:,:,t],cmap='gray', vmin=0, vmax=max_spks)
    ax[7].set_xlim(-0.5,9.5)
    ax[7].set_ylim(-0.5,9.5)
    ax[7].set_title('Inhib V1')
    ax[8].imshow(dact['IV4'][:,:,t],cmap='gray', vmin=0, vmax=max_spks)
    ax[8].set_xlim(-0.5,4.5)
    ax[8].set_ylim(-0.5,4.5)
    ax[8].set_title('Inhib V4')
    ax[9].imshow(dact['IIT'][:,:,t],cmap='gray', vmin=0, vmax=max_spks)
    ax[9].set_xlim(-0.5,2.5)
    ax[9].set_ylim(-0.5,2.5)
    ax[9].set_title('Inhib IT')
    plt.colorbar(pcm2, cax = cbaxes)  
    #plt.colorbar(pcm2)
    plt.pause(2)
    

