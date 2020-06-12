import random
from matplotlib import pyplot as plt
import numpy as np
import gym

nbsteps = 20000

courtXRng = (20, 140)
courtYRng = (34, 194)
racketXRng = (141, 144)

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
  if sIC.shape[0]*sIC.shape[1]==np.shape(Obj_inds)[0]:
    ypos = -1
    xpos = -1
  else:
    ypos = np.median(Obj_inds,0)[0]
    xpos = np.median(Obj_inds,0)[1]
  return xpos, ypos

#env = gym.make('Pong-v0',frameskip=3)
env = gym.make('PongNoFrameskip-v4')
env.reset()

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

observation, reward, done, info = env.step(1)
xpos_Ball2, ypos_Ball2 = findobj (observation, courtXRng, courtYRng)
xpos_Racket2, ypos_Racket2 = findobj (observation, racketXRng, courtYRng)
predY = predictBallRacketYIntercept(xpos_Ball,ypos_Ball,xpos_Ball2,ypos_Ball2)
for _ in range(nbsteps):
  if predY==-1:
    caction = random.randint(3,4)
  else:
    targetY = ypos_Racket2 - predY
    if targetY>8:
      caction = 4
    elif targetY<-8:
      caction = 3
    else:
      caction = 1
  observation, reward, done, info = env.step(caction)
  env.render()
  xpos_Ball = xpos_Ball2
  ypos_Ball = ypos_Ball2
  xpos_Ball2, ypos_Ball2 = findobj (observation, courtXRng, courtYRng)
  xpos_Racket2, ypos_Racket2 = findobj (observation, racketXRng, courtYRng)
  predY = predictBallRacketYIntercept(xpos_Ball,ypos_Ball,xpos_Ball2,ypos_Ball2)
  if done==1:
    env.reset()
