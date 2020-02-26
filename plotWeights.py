import numpy
import matplotlib.pyplot as plt

A = numpy.loadtxt('data/AdjustableWeights.txt')
postN = A[:,2]
postN_MR = numpy.where(postN>1183)
postN_ML = numpy.where(postN<1184)#first one is bad
Mneurons_R = numpy.unique(postN[postN_MR])
Mneurons_L = numpy.unique(postN[postN_ML])

times = A[:,0]
utimes = numpy.unique(times)

#numpy.zeros(shape=(20,20))
All_ML_weights = []
All_MR_weights = []
for t in utimes:
    ctime_inds = numpy.where(times==t)
    ctime_postNs = A[ctime_inds,2]
    ctime_preNs = A[ctime_inds,1]
    ctime_weights = A[ctime_inds,4]
    ML_weights = []
    MR_weights = []
    for n in ctime_postNs[0]:
        cn_preNs = ctime_preNs[0][numpy.where(ctime_postNs[0]==n)]
        if len(cn_preNs)>len(numpy.unique(cn_preNs)):
            print(len(cn_preNs)-len(numpy.unique(cn_preNs)))
        cn_weights = ctime_weights[0][numpy.where(ctime_postNs[0]==n)]       
        if n in Mneurons_R:
            MR_weights.append(numpy.sum(cn_weights))
        elif n in Mneurons_L:
            ML_weights.append(numpy.sum(cn_weights))
    All_ML_weights.append(numpy.sum(ML_weights))
    All_MR_weights.append(numpy.sum(MR_weights))

#plt.subplot(1,2,1)
plt.plot(utimes,All_ML_weights,'r-')
plt.plot(utimes,All_MR_weights,'b-')
plt.xlabel('time [msec]')
plt.legend(('ML','MR'),loc='upper left')
plt.title('Weights')
plt.show()
