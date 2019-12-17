import numpy as np
import matplotlib.pyplot as plt
import pickle

##Change the rasterdata file below
phl_file = open('RasterData.pkl','rb')
data1 = pickle.load(phl_file)
spkTimes = data1["spkTimes"]
spkInds = data1["spkInds"]
cellIDs = data1["cellGids"]
skColors = data1["spkColors"] 

totalDur = 10000
tBin = 100


#NB of excitatory neurons
R = 6400
V1 = 6400
V4 = 1600
IT = 400
#NB of inhibitory neurons
InV1 = 1600
InV4 = 400
InIT = 100
#NB of neurons in Motor cortex
MI = 400
MO = 100
InMI = 100

AllCells = np.array(spkInds)
AllCells_spkTimes = np.array(spkTimes)

ReCells = AllCells[AllCells<6400]
V1eCells =AllCells[(AllCells>6399) & (AllCells<12800)]
V4eCells =AllCells[(AllCells>12799) &(AllCells<14400)]
ITeCells =AllCells[(AllCells>14399) &(AllCells<14800)]
V1iCells =AllCells[(AllCells>14799) &(AllCells<16400)]
V4iCells =AllCells[(AllCells>16399) &(AllCells<16800)]
ITiCells =AllCells[(AllCells>16799) &(AllCells<16900)]

MIeCells =AllCells[(AllCells>16899) &(AllCells<17300)]
MOeCells =AllCells[(AllCells>17299) &(AllCells<17400)]
MIiCells =AllCells[(AllCells>17399) &(AllCells<17500)]

ReCells_spkTimes = AllCells_spkTimes[AllCells<6400]
V1eCells_spkTimes = AllCells_spkTimes[(AllCells>6399) & (AllCells<12800)]
V4eCells_spkTimes = AllCells_spkTimes[(AllCells>12799) &(AllCells<14400)]
ITeCells_spkTimes = AllCells_spkTimes[(AllCells>14399) &(AllCells<14800)]
V1iCells_spkTimes = AllCells_spkTimes[(AllCells>14799) &(AllCells<16400)]
V4iCells_spkTimes = AllCells_spkTimes[(AllCells>16399) &(AllCells<16800)]
ITiCells_spkTimes = AllCells_spkTimes[(AllCells>16799) &(AllCells<16900)]

MIeCells_spkTimes =AllCells_spkTimes[(AllCells>16899) &(AllCells<17300)]
MOeCells_spkTimes =AllCells_spkTimes[(AllCells>17299) &(AllCells<17400)]
MIiCells_spkTimes =AllCells_spkTimes[(AllCells>17399) &(AllCells<17500)]

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
plt.plot(mFR_MIi,color = '0.75')
plt.legend(('R','V1','V4','IT','MI','MO'),loc='upper left')
plt.title('Excitatory populations')

plt.subplot(1,2,2)
plt.plot(mFR_V1i,'b-')
plt.plot(mFR_V4i,'r-')
plt.plot(mFR_ITi,'k-')
plt.plot(mFR_MIi,'m-')
plt.title('Inhibitory populations')
plt.show()

