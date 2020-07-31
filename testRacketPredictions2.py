import random
from matplotlib import pyplot as plt
import numpy as np
import gym
#from pylab import *
nbsteps = 200000

#breakout-specfici 
# courtXRng = (9, 159)
courtXRng = (9,149)
# courtYRng = (32, 189)
courtYRng = (93, 188)
racketYRng = (189,192) 

xpos_Ball = -1 #previous location
ypos_Ball = -1
xpos_Ball2 = -1 #current location
ypos_Ball2 = -1

def findobj (img, xrng, yrng):
  subimg = img[yrng[0]:yrng[1],xrng[0]:xrng[1],:]
  sIC = np.sum(subimg,2)
  pixelVal = np.amax(sIC)
  sIC[sIC<pixelVal]=0
  Obj_inds = []
  for i in range(sIC.shape[0]):
    for j in range(sIC.shape[1]):
      if sIC[i,j]>0:
        Obj_inds.append([i,j])
  if sIC.shape[0]*sIC.shape[1]==np.shape(Obj_inds)[0] or len(Obj_inds)==0: # if number of elements is equal, no sig obj is found
    ypos = -1
    xpos = -1
  else:
    Obj = np.median(Obj_inds,0)
    # print(yrng, type(Obj), type(Obj_inds))
    # print(np.median(Obj_inds,0))
    ypos = np.median(Obj_inds,0)[0]
    xpos = np.median(Obj_inds,0)[1]
  return xpos, ypos

# env = gym.make('Pong-v0',frameskip=3)
env = gym.make('Breakout-v0', frameskip=3)
env.reset()

#For pong (horizontal game)
def predictBallRacketYIntercept(xpos_Ball,ypos_Ball,xpos_Ball2,ypos_Ball2):
  if ((xpos_Ball==-1) or (xpos_Ball2==-1)):
    predY = -1
  else:
    deltax = xpos_Ball2-xpos_Ball
    if deltax<=0:
      predY = -1
    else:
      if ypos_Ball<0:
        predY = -1
      else:
        NB_intercept_steps = np.ceil((120.0 - xpos_Ball2)/deltax)
        deltay = ypos_Ball2-ypos_Ball
        predY_nodeflection = ypos_Ball2 + (NB_intercept_steps*deltay)
        if predY_nodeflection<0:
          predY = -1*predY_nodeflection
        elif predY_nodeflection>160:
          predY = predY_nodeflection-160
        else:
          predY = predY_nodeflection
  return predY

#For breakout (vertical game)
def predictBallRacketXIntercept(xpos1, ypos1, xpos2, ypos2):  
  courtHeight = courtYRng[1] - courtYRng[0]
  courtWidth = courtXRng[1] - courtXRng[0]
  if ((ypos1==-1) or (ypos2==-1)):
    predX = -1
    # print ('Error 1')
  else:
    deltay = ypos2-ypos1
    if deltay<=0:
      predX = -1
      print( deltay, ypos2, ypos1, predX)
      # print ('Error 2')
    else:
      if xpos1<0:
        predX = -1
        # print ('Error 3')
      else:
        NB_intercept_steps = np.ceil((courtHeight - ypos2)/deltay)
        deltax = xpos2-xpos1
        predX_nodeflection = xpos2 + (NB_intercept_steps*deltax)
        if predX_nodeflection<0:
          predX = -1*predX_nodeflection
          # print ('Error 4')
        elif predX_nodeflection>courtWidth:
          predX = predX_nodeflection-courtWidth
          # print ('Error 5')
        else:
          predX = predX_nodeflection
  return predX
observation, reward, done, info = env.step(1)
xpos_Ball2, ypos_Ball2 = findobj (observation, courtXRng, courtYRng)
xpos_Racket2, ypos_Racket2 = findobj (observation, courtXRng, racketYRng)

#breakout-specific
predX = predictBallRacketXIntercept(xpos_Ball,ypos_Ball,xpos_Ball2,ypos_Ball2)

#ion()

for _ in range(nbsteps):
  if predX==-1:
    caction = np.random.randint(2,4) 	# 4 is not included, really pick of 2 and 3
    # print('Random')
  else:
    targetX = xpos_Racket2 - predX
    if targetX>8:
      caction = 3 #left
      # print('Target left')
    elif targetX<-8:
      caction = 2 #right
      # print('Target right')
    else:
      caction = 1 #stay
      # print('Target stay')
  observation, reward, done, info = env.step(caction)
  env.render()
  xpos_Ball = xpos_Ball2
  ypos_Ball = ypos_Ball2
  xpos_Ball2, ypos_Ball2 = findobj (observation, courtXRng, courtYRng)
  xpos_Racket2, ypos_Racket2 = findobj (observation, courtXRng, racketYRng)
  predX = predictBallRacketXIntercept(xpos_Ball,ypos_Ball,xpos_Ball2,ypos_Ball2)
  #imshow(observation,origin='upper'); plot([xpos_Racket2+courtXRng[0]],[predY+courtYRng[0]],'ro')
  if done==1:
    env.reset()
