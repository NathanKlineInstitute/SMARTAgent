import numpy as np
import matplotlib.pyplot as plt
import pickle
from conf import dconf

##Change the rasterdata file below
phl_file = open('data/'+dconf['sim']['name']+'dRasterData.pkl','rb')
data1 = pickle.load(phl_file)
spkTimes = data1["spkTimes"]
spkInds = data1["spkInds"]
cellIDs = data1["cellGids"]
skColors = data1["spkColors"] 

totalDur = 100000
tBin = 1000


#NB of excitatory neurons
R = 400
V1 = 400
V4 = 100
IT = 25
#NB of inhibitory neurons
InR = 100
InV1 = 100
InV4 = 25
InIT = 9
#NB of neurons in Motor cortex
MI = 25
MO = 9
InMI = 9

AllCells = np.array(spkInds)
AllCells_spkTimes = np.array(spkTimes)

ReCells = AllCells[AllCells<R]
V1eCells =AllCells[(AllCells>R-1) & (AllCells<R+V1)]
V4eCells =AllCells[(AllCells>R+V1-1) &(AllCells<R+V1+V4)]
ITeCells =AllCells[(AllCells>R+V1+V4-1) &(AllCells<R+V1+V4+IT)]
RiCells =AllCells[(AllCells>R+V1+V4+IT-1) &(AllCells<R+V1+V4+IT+InR)]
V1iCells =AllCells[(AllCells>R+V1+V4+IT+InR-1) &(AllCells<R+V1+V4+IT+InR+InV1)]
V4iCells =AllCells[(AllCells>R+V1+V4+IT+InR+InV1-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4)]
ITiCells =AllCells[(AllCells>R+V1+V4+IT+InR+InV1+InV4-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4+InIT)]

MIeCells =AllCells[(AllCells>R+V1+V4+IT+InR+InV1+InV4+InIT-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4+InIT+MI)]
MOeCells =AllCells[(AllCells>R+V1+V4+IT+InR+InV1+InV4+InIT+MI-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4+InIT+MI+MO)]
MIiCells =AllCells[(AllCells>R+V1+V4+IT+InR+InV1+InV4+InIT+MI+MO-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4+InIT+MI+MO+InMI)]

ReCells_spkTimes = AllCells_spkTimes[AllCells<R]
V1eCells_spkTimes = AllCells_spkTimes[(AllCells>R-1) & (AllCells<R+V1)]
V4eCells_spkTimes = AllCells_spkTimes[(AllCells>R+V1-1) &(AllCells<R+V1+V4)]
ITeCells_spkTimes = AllCells_spkTimes[(AllCells>R+V1+V4-1) &(AllCells<R+V1+V4+IT)]
RiCells_spkTimes = AllCells_spkTimes[(AllCells>R+V1+V4+IT-1) &(AllCells<R+V1+V4+IT+InR)]
V1iCells_spkTimes = AllCells_spkTimes[(AllCells>R+V1+V4+IT+InR-1) &(AllCells<R+V1+V4+IT+InR+InV1)]
V4iCells_spkTimes = AllCells_spkTimes[(AllCells>R+V1+V4+IT+InR+InV1-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4)]
ITiCells_spkTimes = AllCells_spkTimes[(AllCells>R+V1+V4+IT+InR+InV1+InV4-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4+InIT)]

MIeCells_spkTimes =AllCells_spkTimes[(AllCells>R+V1+V4+IT+InR+InV1+InV4+InIT-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4+InIT+MI)]
MOeCells_spkTimes =AllCells_spkTimes[(AllCells>R+V1+V4+IT+InR+InV1+InV4+InIT+MI-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4+InIT+MI+MO)]
MIiCells_spkTimes =AllCells_spkTimes[(AllCells>R+V1+V4+IT+InR+InV1+InV4+InIT+MI+MO-1) &(AllCells<R+V1+V4+IT+InR+InV1+InV4+InIT+MI+MO+InMI)]

def computeMeanFiringRate(totalDur, tBin, Cells, Cells_spkTimes,NBCells):
    uniqueCells = np.unique(Cells)
    nbBins = int(totalDur/tBin)
    spkCount = np.zeros((nbBins,len(uniqueCells)))
    Cells_spkCount = []
    totalSpikes = []
    count = 0
    for cell in uniqueCells:
        cCell_spkTimes = Cells_spkTimes[Cells==cell]
        Cells_spkCount.append(count)
        totalSpikes.append(len(cCell_spkTimes))
        for n in range(nbBins):
            t0 = n*tBin
            t1 = ((n+1)*tBin) - 1
            spk_times = cCell_spkTimes[(cCell_spkTimes>t0-0.001) & (cCell_spkTimes<t1+0.01)]
            spkCount[n][count]=len(spk_times)
        count = count+1
    mfactor = 1000/(tBin*NBCells) 
    meanFiringRate = mfactor*np.sum(spkCount, axis = 1)
    return meanFiringRate

mFR_Re = computeMeanFiringRate(totalDur, tBin, ReCells, ReCells_spkTimes,R)
mFR_V1e = computeMeanFiringRate(totalDur, tBin, V1eCells, V1eCells_spkTimes,V1)
mFR_V4e = computeMeanFiringRate(totalDur, tBin, V4eCells, V4eCells_spkTimes,V4)
mFR_ITe = computeMeanFiringRate(totalDur, tBin, ITeCells, ITeCells_spkTimes,IT)
mFR_MIe = computeMeanFiringRate(totalDur, tBin, MIeCells, MIeCells_spkTimes,MI)
mFR_MOe = computeMeanFiringRate(totalDur, tBin, MOeCells, MOeCells_spkTimes,MO)

mFR_Ri = computeMeanFiringRate(totalDur, tBin, RiCells, RiCells_spkTimes,InR)
mFR_V1i = computeMeanFiringRate(totalDur, tBin, V1iCells, V1iCells_spkTimes,InV1)
mFR_V4i = computeMeanFiringRate(totalDur, tBin, V4iCells, V4iCells_spkTimes,InV4)
mFR_ITi = computeMeanFiringRate(totalDur, tBin, ITiCells, ITiCells_spkTimes,InIT)
mFR_MIi = computeMeanFiringRate(totalDur, tBin, MIiCells, MIiCells_spkTimes,InMI)

plt.subplot(1,2,1)
plt.plot(mFR_Re,'g-')
plt.plot(mFR_V1e,'b-')
plt.plot(mFR_V4e,'r-')
plt.plot(mFR_ITe,'k-')
plt.plot(mFR_MIe,'m-')
plt.plot(mFR_MOe,color = '0.75')
plt.legend(('R','V1','V4','IT','MI','MO'),loc='upper left')
plt.title('Excitatory populations')

plt.subplot(1,2,2)
plt.plot(mFR_Ri,'g-')
plt.plot(mFR_V1i,'b-')
plt.plot(mFR_V4i,'r-')
plt.plot(mFR_ITi,'k-')
plt.plot(mFR_MIi,'m-')
plt.title('Inhibitory populations')
plt.show()

