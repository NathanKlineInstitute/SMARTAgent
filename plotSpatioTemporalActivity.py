import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

Input_Images = np.loadtxt('InputImages.txt')
New_InputImages = []
NB_Images = int(Input_Images.shape[0]/Input_Images.shape[1])
for x in range(NB_Images):
    fp = x*Input_Images.shape[1]
    cImage = Input_Images[fp:fp+20,:]
    New_InputImages.append(cImage)
New_InputImages = np.array(New_InputImages)

##Change the rasterdata file below
phl_file = open('RasterData.pkl','rb')
data1 = pickle.load(phl_file)
spkTimes = data1["spkTimes"]
spkInds = data1["spkInds"]
cellIDs = data1["cellGids"]
skColors = data1["spkColors"] 

totalDur = 1000
tBin_Size = 20

neuronIDs = np.unique(spkInds)

#NB of excitatory neurons
R = 400
V1 = 400
V4 = 100
IT = 25
#NB of inhibitory neurons
InV1 = 100
InV4 = 25
InIT = 9
#NB of neurons in Motor cortex
MI = 25
MO = 9
InMI = 9

AllCells = np.array(spkInds)
AllCells_spkTimes = np.array(spkTimes)

lastR = R
lastV1 = R+V1
lastV4 = R+V1+V4
lastIT = R+V1+V4+IT
lastInV1 = R+V1+V4+IT+InV1
lastInV4 = R+V1+V4+IT+InV1+InV4
lastInIT = R+V1+V4+IT+InV1+InV4+InIT
lastMI = R+V1+V4+IT+InV1+InV4+InIT+MI
lastMO = R+V1+V4+IT+InV1+InV4+InIT+MI+MO
lastInMI = R+V1+V4+IT+InV1+InV4+InIT+MI+MO+InMI 

ReCells = AllCells[AllCells<lastR]
V1eCells =AllCells[(AllCells>lastR-1) & (AllCells<lastV1)]
V4eCells =AllCells[(AllCells>lastV1-1) &(AllCells<lastV4)]
ITeCells =AllCells[(AllCells>lastV4-1) &(AllCells<lastIT)]
V1iCells =AllCells[(AllCells>lastIT-1) &(AllCells<lastInV1)]
V4iCells =AllCells[(AllCells>lastInV1-1) &(AllCells<lastInV4)]
ITiCells =AllCells[(AllCells>lastInV4-1) &(AllCells<lastInIT)]

MIeCells =AllCells[(AllCells>lastInIT-1) &(AllCells<lastMI)]
MOeCells =AllCells[(AllCells>lastMI-1) &(AllCells<lastMO)]
MIiCells =AllCells[(AllCells>lastMO-1) &(AllCells<lastInMI)]

#Re-index cell ids
RCells = np.subtract(ReCells,np.min(ReCells))
V1Cells = np.subtract(V1eCells,np.min(V1eCells))
V4Cells = np.subtract(V4eCells,np.min(V4eCells))
ITCells = np.subtract(ITeCells,np.min(ITeCells))
InV1Cells = np.subtract(V1iCells,np.min(V1iCells))
InV4Cells = np.subtract(V4iCells,np.min(V4iCells))
InITCells = np.subtract(ITiCells,np.min(ITiCells))
MICells = np.subtract(MIeCells,np.min(MIeCells))
MOCells = np.subtract(MOeCells,np.min(MOeCells))
InMICells = np.subtract(MIiCells,np.min(MIiCells))

ReCells_spkTimes = AllCells_spkTimes[AllCells<lastR]
V1eCells_spkTimes = AllCells_spkTimes[(AllCells>lastR-1) & (AllCells<lastV1)]
V4eCells_spkTimes = AllCells_spkTimes[(AllCells>lastV1-1) &(AllCells<lastV4)]
ITeCells_spkTimes = AllCells_spkTimes[(AllCells>lastV4-1) &(AllCells<lastIT)]
V1iCells_spkTimes = AllCells_spkTimes[(AllCells>lastIT-1) &(AllCells<lastInV1)]
V4iCells_spkTimes = AllCells_spkTimes[(AllCells>lastInV1-1) &(AllCells<lastInV4)]
ITiCells_spkTimes = AllCells_spkTimes[(AllCells>lastInV4-1) &(AllCells<lastInIT)]

MIeCells_spkTimes =AllCells_spkTimes[(AllCells>lastInIT-1) &(AllCells<lastMI)]
MOeCells_spkTimes =AllCells_spkTimes[(AllCells>lastMI-1) &(AllCells<lastMO)]
MIiCells_spkTimes =AllCells_spkTimes[(AllCells>lastMO-1) &(AllCells<lastInMI)]

t1 = range(0,totalDur,tBin_Size)
t2 = range(tBin_Size,totalDur+tBin_Size,tBin_Size)

def generateActivityMap(t1, t2, NCells, NCells_spkTimes):
    N = len(np.unique(NCells))
    Nact = np.zeros(shape=(int(np.sqrt(N)),int(np.sqrt(N)),len(t1)))
    for i in range(int(np.sqrt(N))):
        for j in range(int(np.sqrt(N))):
            cNeuronID = j+(i*int(np.sqrt(N)))
            cNeuron_spkt = NCells_spkTimes[NCells==cNeuronID]
            for t in range(len(t1)):
                cbinSpikes = cNeuron_spkt[(cNeuron_spkt>t1[t]) & (cNeuron_spkt<=t2[t])]
                Nact[i][j][t] = len(cbinSpikes)
    return Nact

Ract = generateActivityMap(t1, t2, RCells, ReCells_spkTimes)
V1act = generateActivityMap(t1, t2, V1Cells, V1eCells_spkTimes)
V4act = generateActivityMap(t1, t2, V4Cells, V4eCells_spkTimes)
ITact = generateActivityMap(t1, t2, ITCells, ITeCells_spkTimes)

IV1act = generateActivityMap(t1, t2, InV1Cells, V1iCells_spkTimes)
IV4act = generateActivityMap(t1, t2, InV4Cells, V4iCells_spkTimes)
IITact = generateActivityMap(t1, t2, InITCells, ITiCells_spkTimes)


############################

#plt.axis('off')

fig, axs = plt.subplots(2, 4)
ax = axs.ravel()

cbaxes = fig.add_axes([0.95, 0.4, 0.01, 0.2]) 
#cb = plt.colorbar(ax1, cax = cbaxes)  

for t in range(len(t1)):
    fig.suptitle('20 ms binned activity' + str(t*tBin_Size) + ' ms')
    ax[0].imshow(Ract[:,:,t],cmap='gray', vmin=0, vmax=2)
    ax[0].set_title('Excit R')
    ax[0].set_xlim(-0.5,19.5)
    ax[0].set_ylim(-0.5,19.5)
    pcm2 = ax[1].imshow(V1act[:,:,t],cmap='gray', vmin=0, vmax=2)
    ax[1].set_xlim(-0.5,19.5)
    ax[1].set_ylim(-0.5,19.5)
    ax[1].set_title('Excit V1')
    ax[2].imshow(V4act[:,:,t],cmap='gray', vmin=0, vmax=2)
    ax[2].set_xlim(-0.5,9.5)
    ax[2].set_ylim(-0.5,9.5)
    ax[2].set_title('Excit V4')
    ax[3].imshow(ITact[:,:,t],cmap='gray', vmin=0, vmax=2)
    ax[3].set_xlim(-0.5,4.5)
    ax[3].set_ylim(-0.5,4.5)
    ax[3].set_title('Excit IT')
    ax[4].axis('off')
    ax[4].imshow(New_InputImages[t,:,:],cmap='gray', vmin=0, vmax = 255)
    ax[4].set_xlim(-0.5,19.5)
    ax[4].set_ylim(-0.5,19.5)
    ax[4].set_title('Input Images')
    ax[5].imshow(IV1act[:,:,t],cmap='gray', vmin=0, vmax=2)
    ax[5].set_xlim(-0.5,9.5)
    ax[5].set_ylim(-0.5,9.5)
    ax[5].set_title('Inhib V1')
    ax[6].imshow(IV4act[:,:,t],cmap='gray', vmin=0, vmax=2)
    ax[6].set_xlim(-0.5,4.5)
    ax[6].set_ylim(-0.5,4.5)
    ax[6].set_title('Inhib V4')
    ax[7].imshow(IITact[:,:,t],cmap='gray', vmin=0, vmax=2)
    ax[7].set_xlim(-0.5,2.5)
    ax[7].set_ylim(-0.5,2.5)
    ax[7].set_title('Inhib IT')
    plt.colorbar(pcm2, cax = cbaxes)  
    #plt.colorbar(pcm2)
    plt.pause(2)
    

