import numpy
import matplotlib.pyplot as plt

A = numpy.loadtxt('data/AdjustableWeights.txt')
postN = A[1,:]
postN_MR = numpy.where(postN>1183)
postN_ML = numpy.where(postN<1184)#first one is bad

Mneurons_R = postN[postN_MR]
Mneurons_L = postN[postN_ML[0][1:len(postN_ML[0])]]

time = A[3:A.shape[0],0]

Weights = A[3:A.shape[0]]
Weights_MR = Weights[:,postN_MR[0]]
Weights_ML = Weights[:,postN_ML[0][1:len(postN_ML[0])]]

totalWeight_MR = numpy.sum(Weights_MR,1)
totalWeight_ML = numpy.sum(Weights_ML,1)

#plt.subplot(1,2,1)
plt.plot(time,totalWeight_MR,'r-')
plt.plot(time,totalWeight_ML,'b-')
plt.xlabel('time [msec]')
plt.legend(('MR','ML'),loc='upper left')
plt.title('Weights')
plt.show()
