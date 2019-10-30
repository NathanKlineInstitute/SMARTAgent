from matplotlib import pyplot as plt
import random
import numpy

import gym
env = gym.make("Pong-v0")
env.reset()
#env.render()
#dsum_Images = numpy.zeros(shape=(40,40))
count = 0
for _ in range(20):
  dsum_Images = numpy.zeros(shape=(80,80))
  for _ in range(5):
    action = random.randint(3,4)
    observation, reward, done, info = env.step(action)
    Image = observation[34:194][:][:]
    gray_Image = numpy.zeros(shape=(160,160))
    for i in range(160):
      for j in range(160):
        gray_Image[i][j]= 0.2989*Image[i][j][0] + 0.5870*Image[i][j][1] + 0.1140*Image[i][j][2]
    gray_ds = gray_Image[:,range(0,gray_Image.shape[1],2)]
    gray_ds = gray_ds[range(0,gray_ds.shape[0],2),:]
    gray_ds = numpy.where(gray_ds<127,0,gray_ds)
    gray_ds = numpy.where(gray_ds>=127,255,gray_ds)
    if count==0:
      i0 = 0.4*gray_ds
      plt.subplot(2,3,1)
      plt.imshow(i0, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
      plt.title('t0')
      count = count+1
    elif count==1:
      i1 = 0.55*gray_ds
      plt.subplot(2,3,2)
      plt.imshow(i1, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
      plt.title('t1')
      count = count+1
    elif count==2:
      i2 = 0.7*gray_ds
      plt.subplot(2,3,3)
      plt.imshow(i2, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
      plt.title('t2')
      count = count+1
    elif count==3:
      i3 = 0.85*gray_ds
      plt.subplot(2,3,4)
      plt.imshow(i3, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
      plt.title('t3')
      count = count+1
    else:
      i4 = 1.0*gray_ds
      plt.subplot(2,3,5)
      plt.imshow(i4, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
      plt.title('t4')
      count = 0
  dsum_Images = numpy.maximum(i0,i1)
  dsum_Images = numpy.maximum(dsum_Images,i2)
  dsum_Images = numpy.maximum(dsum_Images,i3)
  dsum_Images = numpy.maximum(dsum_Images,i4)
  plt.subplot(2,3,6)
  plt.imshow(dsum_Images, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
  plt.title('max[t0+t1+t2+t3+t4]')
  plt.show()
  #CONVERT PIXEL INTENSITIES INTO FIRING RATES
  fr_Images = 40/(1+numpy.exp((numpy.multiply(-1,dsum_Images)+123)/25))
#  plt.subplot(1,2,2)
#  plt.imshow(fr_Images, cmap=plt.get_cmap('gray'), vmin=0, vmax = 55)
#  plt.hist2d(fr_Images)
#  plt.show()
#  print(numpy.histogram(dsum_Images[dsum_Images > 0]))
#  print(numpy.histogram(fr_Images[fr_Images > 1]))
