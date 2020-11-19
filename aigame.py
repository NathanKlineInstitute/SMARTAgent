"""
AIGame: connects OpenAI gym game to the model (V1-M1-RL)
Adapted from arm.py
Original Version: 2015jan28 by salvadordura@gmail.com
Modified Version: 2019oct1 by haroon.anwar@gmail.com
Modified 2019-2020 samn
Modified 2020 davidd
"""

from neuron import h
from pylab import concatenate, figure, show, ion, ioff, pause,xlabel, ylabel, plot, Circle, sqrt, arctan, arctan2, close
from copy import copy, deepcopy
from random import uniform, seed, sample, randint
from matplotlib import pyplot as plt
import random
import numpy as np
from skimage.transform import downscale_local_mean, resize
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import json
import gym
import sys
from gym import wrappers
from time import time
from collections import OrderedDict
from imgutils import getoptflow
from imgutils import getObjectsBoundingBoxes, getObjectMotionDirection
import cv2
from centroidtracker import CentroidTracker
from conf import dconf

# make the environment - env is global so that it only gets created on a single node (important when using MPI with > 1 node)
useSimulatedEnv = False
try:
  if 'useSimulatedEnv' in dconf: useSimulatedEnv = dconf['useSimulatedEnv']
  if useSimulatedEnv:
    from simulatePong import simulatePong
    pong = simulatePong()
  else:
    if 'frameskip' in dconf['env']:
      env = gym.make(dconf['env']['name'],frameskip=dconf['env']['frameskip'],repeat_action_probability=0.)
    else:
      env = gym.make(dconf['env']['name'],repeat_action_probability=0.)    
    if dconf['env']['savemp4']: env = wrappers.Monitor(env, './videos/' + dconf['sim']['name'] + '/',force=True)
    env.reset()
except:
  print('Exception in makeENV')
  env = gym.make('Pong-v0',framepskip=1,repeat_action_probability=0.)
  env = wrappers.Monitor(env, './videos/' + str(time()) + '/',force=True)
  env.reset()

# get smallest angle difference
def getangdiff (ang1, ang2):
  if ang1 > 180.0:
    ang1 -= 360.0
  if ang2 > 180.0:
    ang2 -= 360.0
  angdiff = ang1 - ang2
  if angdiff > 180.0:
    angdiff-=360.0
  elif angdiff < -180.0:
    angdiff+=360.0
  return angdiff
  
class AIGame:
  """ Interface to OpenAI gym game 
  """
  def __init__ (self,fcfg='sim.json'): # initialize variables
    if useSimulatedEnv: self.pong = pong
    else: self.env = env
    self.countAll = 0
    self.ldir = ['E','NE','N', 'NW','W','SW','S','SE']
    self.ldirpop = ['EV1D'+Dir for Dir in self.ldir]
    if dconf['net']['allpops']['ER']>0: self.InputPop = 'ER'
    else: self.InputPop = 'EV1'    
    self.lratepop = [self.InputPop] # populations that we calculate rates for
    for pop in self.ldirpop: self.lratepop.append(pop)
    self.reducedNet = dconf['sim']['useReducedNetwork'] # reduced architecture with associations
    self.dReceptiveField = OrderedDict({pop:np.amax(list(dconf['net']['alltopolconvcons'][pop].values())) for pop in self.lratepop})
    self.dInputs = OrderedDict({pop:int((np.sqrt(dconf['net']['allpops'][pop])+self.dReceptiveField[pop]-1)**2) for pop in self.lratepop})
    #self.dFVec = OrderedDict({pop:h.Vector() for pop in self.lratepop}) # NEURON Vectors for firing rate calculations
    self.dFiringRates = OrderedDict({pop:np.zeros(dconf['net']['allpops'][pop]) for pop in self.lratepop})# python objects for firing rate calculations
    if dconf['net']['useNeuronPad']:
      for pop in self.dInputs.keys():
        self.dFiringRates[pop] = np.zeros(self.dInputs[pop])
    self.dAngPeak = OrderedDict({'EV1DE': 0.0,'EV1DNE': 45.0, # receptive field peak angles for the direction selective populations
                                'EV1DN': 90.0,'EV1DNW': 135.0,
                                'EV1DW': 180.0,'EV1DSW': 225.0,
                                'EV1DS': 270.0,'EV1DSE': 315.0})
    self.AngRFSigma = dconf['net']['AngRFSigma']
    self.AngRFSigma2 = dconf['net']['AngRFSigma']**2 # angular receptive field (RF) sigma squared used for dir selective neuron RFs
    if self.AngRFSigma2 <= 0.0: self.AngRFSigma2=1.0
    self.EXPDir = dconf['net']['EXPDir']
    if dconf['net']['useNeuronPad']:
      self.input_dim = int(np.sqrt(self.dInputs[self.InputPop]))
      self.objExtension = 'Vertical'  # this could be included in sim.json....
    else:  
      self.input_dim = int(np.sqrt(dconf['net']['allpops'][self.InputPop])) # input image XY plane width,height -- not used anywhere    
    self.locationNeuronRate = dconf['net']['LocMaxRate']
    if self.reducedNet:
      self.dirSensitiveNeuronDim = 20 #Assuming the downscaling factor is 8.
    else:
      self.dirSensitiveNeuronDim = int(np.sqrt(dconf['net']['allpops']['EV1DE'])) # direction sensitive neuron XY plane width,height
    self.dirSensitiveNeuronRate = (dconf['net']['DirMinRate'], dconf['net']['DirMaxRate']) # min, max firing rate (Hz) for dir sensitive neurons
    self.FiringRateCutoff = dconf['net']['FiringRateCutoff']
    self.intaction = int(dconf['actionsPerPlay']) # integrate this many actions together before returning reward information to model
    # these are Pong-specific coordinate ranges; should later move out of this function into Pong-specific functions
    self.courtYRng = (34, 194) # court y range
    self.racket0XRng = (16, 20)
    self.courtXRng = (20, 140) # court x range
    self.racketXRng = (140, 144) # racket x range
    self.dObjPos = {'time':[], 'racket':[], 'ball':[]}
    self.last_obs = [] # previous observation
    self.last_ball_dir = 0 # last ball direction
    self.FullImages = [] # full resolution images from game environment
    self.ReducedImages = [] # low resolution images from game environment used as input to neuronal network model
    self.ldflow = [] # list of dictionary of optical flow (motion) fields
    if dconf['sim']['saveAssignedFiringRates']:
      self.dAllFiringRates = []
    if dconf['DirectionDetectionAlgo']['CentroidTracker']:
      self.ct = CentroidTracker()
      self.objects = OrderedDict() # objects detected in current frame
      self.last_objects = OrderedDict() # objects detected in previous frame
    self.stayStepLim = dconf['stayStepLim'] # number of steps to hold still after every move (to reduce momentum)
    # Note that takes 6 stays instead of 3 because it seems every other input is ignored (check dad_notes.txt for details)
    # however, using stayStepLim > 0 means model behavior is slower
    self.downsampshape = (8,8) # default is 20x20 (20 = 1/8 of 160)
    self.racketH = 16 #3 # racket height in pixels
    self.maxYPixel = 160 - 1 # 20 - 1
    self.avoidStuck = False
    self.avoidStuck = dconf['avoidStuck']
    self.useImagePadding = dconf['useImagePadding'] # make sure the number of neurons is correct
    self.padPixelEachSide = 16 # keeping it fix for maximum number of pixels for racket.
    if dconf['net']['allpops'][self.InputPop] == 1600 or dconf['net']['allpops'][self.InputPop] == 2304 or dconf['net']['allpops'][self.InputPop] == 80 or dconf['net']['allpops'][self.InputPop] == 96: # this is for 40x40 (40 = 1/4 of 160)
      self.downsampshape = (4,4)
      #self.racketH *= 2
      #self.maxYPixel = 40 - 1
    elif dconf['net']['allpops'][self.InputPop] == 6400 or dconf['net']['allpops'][self.InputPop] == 9216 or dconf['net']['allpops'][self.InputPop] == 160 or dconf['net']['allpops'][self.InputPop] == 192: # this is for 80x80 (1/2 resolution)
      self.downsampshape = (2,2)
      #self.racketH *= 4
      #self.maxYPixel = 80 - 1      
    elif dconf['net']['allpops'][self.InputPop] == 25600 or dconf['net']['allpops'][self.InputPop] == 36864 or dconf['net']['allpops'][self.InputPop] == 320 or dconf['net']['allpops'][self.InputPop] == 384: # this is for 160x160 (full resolution)
      self.downsampshape = (1,1)
      #self.racketH *= 8
      #self.maxYPixel = 160 - 1
    self.thresh = 100 # 140
    self.binary_Image = None

  def getThreshold (self, I):
    return np.amin(I)+0.1
    #return self.thresh
    #return threshold_otsu(I) 

  def updateInputRatesWithPadding (self, dsum_Images):
    # update input rates to retinal neurons
    tmp_padded_Image = np.amin(dsum_Images)*np.ones(shape=(self.input_dim,self.input_dim))
    padded_Image = np.amin(dsum_Images)*np.ones(shape=(self.input_dim,self.input_dim))
    offset = int((self.dReceptiveField[self.InputPop]-1)/2)
    tmp_padded_Image[offset:offset+dsum_Images.shape[0],offset:offset+dsum_Images.shape[1]]=dsum_Images
    padded_Image[offset:offset+dsum_Images.shape[0],offset:offset+dsum_Images.shape[1]]=dsum_Images
    # find the indices of padded pixels
    paddingInds = []
    for j in range(self.input_dim):
      for i in range(offset):
        paddingInds.append([i,j])
        paddingInds.append([j,i])
      for i in range(offset+dsum_Images.shape[0],self.input_dim):
        paddingInds.append([i,j])
        paddingInds.append([j,i])
    if self.objExtension=='Horizontal':
      for i in range(np.shape(paddingInds)[0]):
        if paddingInds[i][1]<offset:
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0],paddingInds[i][1]:paddingInds[i][1]+offset+1])
        elif paddingInds[i][1]>offset+19:
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0],paddingInds[i][1]-offset:paddingInds[i][1]])
        else:
          padded_Image[paddingInds[i][0],paddingInds[i][1]] = tmp_padded_Image[paddingInds[i][0],paddingInds[i][1]]
    elif self.objExtension=='Vertical':
      for i in range(np.shape(paddingInds)[0]):
        if paddingInds[i][0]<offset:
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0]:paddingInds[i][0]+offset+1,paddingInds[i][1]])
        elif paddingInds[i][0]>offset+19:
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0]-offset:paddingInds[i][0],paddingInds[i][1]])
        else:
          padded_Image[paddingInds[i][0],paddingInds[i][1]] = tmp_padded_Image[paddingInds[i][0],paddingInds[i][1]]
    else:
      for i in range(np.shape(paddingInds)[0]):
        if (paddingInds[i][0]<offset+1) and (paddingInds[i][1]<offset+1):
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0]:paddingInds[i][0]+offset+1,paddingInds[i][1]:paddingInds[i][1]+offset+1])
        elif (paddingInds[i][0]>offset+19) and (paddingInds[i][1]>offset+19):
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0]-offset:paddingInds[i][0],paddingInds[i][0]-offset:paddingInds[i][1]])
        elif (paddingInds[i][0]<offset) and (paddingInds[i][1]>=offset):
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0]:paddingInds[i][0]+offset+1,paddingInds[i][1]-offset:paddingInds[i][1]])
        elif (paddingInds[i][0]>offset+19) and (paddingInds[i][1]<=offset+19):
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0]-offset:paddingInds[i][0],paddingInds[i][1]:paddingInds[i][0]+offset+1])
        elif (paddingInds[i][0]>offset) and (paddingInds[i][1]<offset):
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0],paddingInds[i][1]:paddingInds[i][1]+offset+1])
        elif (paddingInds[i][0]<=offset+19) and (paddingInds[i][1]>offset+19):
          padded_Image[paddingInds[i][0],paddingInds[i][1]]=np.amax(tmp_padded_Image[paddingInds[i][0],paddingInds[i][0]-offset:paddingInds[i][1]])
        else:
          padded_Image[paddingInds[i][0],paddingInds[i][1]] = tmp_padded_Image[paddingInds[i][0],paddingInds[i][1]]
    if dconf['net']['useBinaryImage']:
      self.binary_Image = binary_Image  = padded_Image > self.getThreshold(padded_Image)
      fr_Images = self.locationNeuronRate*binary_Image
    else:
      padded_Image = padded_Image - np.amin(padded_Image)
      padded_Image = (255.0/np.amax(padded_Image))*padded_Image # this will make sure that padded_Image spans 0-255
      fr_Images = self.locationNeuronRate/(1+np.exp((np.multiply(-1,padded_Image)+123)/10))
      #fr_Images = np.subtract(fr_Images,np.min(fr_Images)) #baseline firing rate subtraction. Instead all excitatory neurons are firing at 5Hz.
      #print(np.amin(fr_Images),np.amax(fr_Images))
    self.dFiringRates[self.InputPop] = np.reshape(fr_Images,self.dInputs[self.InputPop]) #400 for 20*20, 900 for 30*30, etc.

  def getNewCoords(self):
    if self.downsampshape[0]==8:
      if self.useImagePadding:
        ds_courtXRng = (5, 19) # court x range
        ds_racketXRng = (19, 20) # racket x range... when used in an image
        ds_racket0XRng = (4, 5)
      else:
        ds_courtXRng = (3, 17) # court x range
        ds_racketXRng = (17, 18) # racket x range... when used in an image
        ds_racket0XRng = (2, 3)
    elif self.downsampshape[0]==4:
      if self.useImagePadding:
        ds_courtXRng = (9, 39) # court x range
        ds_racketXRng = (39, 40) # racket x range... when used in an image
        ds_racket0XRng = (8, 9)  
      else:
        ds_courtXRng = (5, 35) # court x range
        ds_racketXRng = (35, 36) # racket x range... when used in an image
        ds_racket0XRng = (4, 5)
    elif self.downsampshape[0]==2:
      if self.useImagePadding:
        ds_courtXRng = (18, 78)
        ds_racketXRng = (78, 80)
        ds_racket0XRng = (16, 18)  
      else:
        ds_courtXRng = (10, 70) # court x range
        ds_racketXRng = (70, 72) # racket x range... when used in an image
        ds_racket0XRng = (8, 10)
    elif self.downsampshape[0]==1:
      if self.useImagePadding:
        ds_courtXRng = (36, 156) # court x range
        ds_racketXRng = (156,160) # racket x range... when used in an image
        ds_racket0XRng = (32, 36)
      else:
        ds_courtXRng = self.courtXRng # court x range
        ds_racketXRng = self.racketXRng # racket x range... when used in an image
        ds_racket0XRng = self.racket0XRng
    return ds_courtXRng, ds_racket0XRng, ds_racketXRng

  def updateInputRates (self, dsum_Images):
    # update input rates to retinal neurons
    #fr_Images = np.where(dsum_Images>1.0,100,dsum_Images) #Using this to check what number would work for firing rate
    #fr_Images = np.where(dsum_Images<10.0,0,dsum_Images)
    ds_courtXRng, ds_racket0XRng, ds_racketXRng = self.getNewCoords() #racket0 is for the GAME racket
    if self.reducedNet:
      fr_Images = []
      if dconf['net']['useBinaryImage']:
        self.binary_Image = binary_Image = dsum_Images > self.getThreshold(dsum_Images)
        if dconf['sim']['captureTwoObjs']:
          fr_Images = np.zeros(shape=(dsum_Images.shape[0],2))
          fr_Images[:,0] = self.locationNeuronRate*np.amax(binary_Image[:,ds_courtXRng[0]:ds_courtXRng[1]],1)
          fr_Images[:,1] = self.locationNeuronRate*np.amax(binary_Image[:,ds_racketXRng[0]:ds_racketXRng[1]],1)
          #print(fr_Images)        
        else:
          fr_Images = np.zeros(shape=(dsum_Images.shape[0],3))
          fr_Images[:,0] = self.locationNeuronRate*np.amax(binary_Image[:,ds_courtXRng[0]:ds_courtXRng[1]],1)
          fr_Images[:,1] = self.locationNeuronRate*np.amax(binary_Image[:,ds_racketXRng[0]:ds_racketXRng[1]],1)
          fr_Images[:,2] = self.locationNeuronRate*np.amax(binary_Image[:,ds_racket0XRng[0]:ds_racket0XRng[1]],1)
      else:
        dsum_Images = dsum_Images - np.amin(dsum_Images)
        dsum_Images = (255.0/np.amax(dsum_Images))*dsum_Images
        fr_Image = self.locationNeuronRate/(1+np.exp((np.multiply(-1,dsum_Images)+123)/10))
        if dconf['sim']['captureTwoObjs']:
          fr_Images = np.zeros(shape=(dsum_Images.shape[0],2))
          fr_Images[:,0] = np.sum(fr_Image[:,ds_courtXRng[0]:ds_courtXRng[1]],1) # ball y loc
          fr_Images[:,1] = np.sum(fr_Image[:,ds_racketXRng[0]:ds_racketXRng[1]],1) # model racket y loc
        else:
          fr_Images = np.zeros(shape=(dsum_Images.shape[0],4))
          fr_Images[:,0] = np.sum(fr_Image[:,ds_courtXRng[0]:ds_courtXRng[1]],1) # ball y loc
          fr_Images[:,1] = np.sum(fr_Image[:,ds_racketXRng[0]:ds_racketXRng[1]],1) # model-racket y loc
          fr_Images[:,2] = np.sum(fr_Image[:,ds_racket0XRng[0]:ds_racket0XRng[1]],0) # game-racket y loc
    else:
      if dconf['net']['useBinaryImage']:
        self.binary_Image = binary_Image = dsum_Images > self.getThreshold(dsum_Images)
        fr_Images = self.locationNeuronRate*binary_Image
      else:
        dsum_Images = dsum_Images - np.amin(dsum_Images)
        dsum_Images = (255.0/np.amax(dsum_Images))*dsum_Images
        fr_Images = self.locationNeuronRate/(1+np.exp((np.multiply(-1,dsum_Images)+123)/10))
        #fr_Images = np.subtract(fr_Images,np.min(fr_Images)) #baseline firing rate subtraction. Instead all excitatory neurons are firing at 5Hz.
        #print(np.amax(fr_Images))
    self.dFiringRates[self.InputPop] = np.reshape(fr_Images,dconf['net']['allpops'][self.InputPop]) #400 for 20*20, 900 for 30*30, etc.

  def computeMotionFields (self, UseFull=False):
    # compute and store the motion fields and associated data
    if UseFull:
      limage = self.FullImages
    else:
      limage = self.ReducedImages
    if len(limage) < 2:
      flow = np.zeros(shape=(limage[-1].shape[0],limage[-1].shape[1],2))
      mag = np.zeros(shape=(limage[-1].shape[0],limage[-1].shape[1]))
      ang = np.zeros(shape=(limage[-1].shape[0],limage[-1].shape[1]))
      ang[mag == 0] = -100
      goodInds = np.zeros(shape=(limage[-1].shape[0],limage[-1].shape[1]))
      self.ldflow.append({'flow':flow,'mag':mag,'ang':ang,'goodInds':goodInds})
    else:
      self.ldflow.append(getoptflow(limage[-2],limage[-1]))

  def computeAllObjectsMotionDirections(self, UseFull=False):
    #Detect the objects, and initialize the list of bounding box rectangles
    if len(self.FullImages)==0: return
    if UseFull:
      cimage = self.FullImages[-1]
    else:
      cimage = self.ReducedImages[-1]
    rects = getObjectsBoundingBoxes(cimage)
    cimage = np.ascontiguousarray(cimage, dtype=np.uint8)
    # update our centroid tracker using the computed set of bounding box rectangles
    self.objects = self.ct.update(rects)
    if len(self.last_objects)==0: 
      self.last_objects = deepcopy(self.objects)
      flow = np.zeros(shape=(self.dirSensitiveNeuronDim,self.dirSensitiveNeuronDim,2))
      mag = np.zeros(shape=(self.dirSensitiveNeuronDim,self.dirSensitiveNeuronDim))
      ang = np.zeros(shape=(self.dirSensitiveNeuronDim,self.dirSensitiveNeuronDim))
      ang[mag == 0] = -100
      goodInds = np.zeros(shape=(self.dirSensitiveNeuronDim,self.dirSensitiveNeuronDim))
    else:
      dirX, dirY = getObjectMotionDirection(self.objects, self.last_objects, rects, dims=np.shape(cimage)[0],\
                                            FlowWidth=dconf['DirectionDetectionAlgo']['FlowWidth'])
      if np.shape(cimage)[0] != self.dirSensitiveNeuronDim or np.shape(cimage)[1] != self.dirSensitiveNeuronDim:
        dirX = resize(dirX, (self.dirSensitiveNeuronDim, self.dirSensitiveNeuronDim), anti_aliasing=True)
        dirY = resize(dirY, (self.dirSensitiveNeuronDim, self.dirSensitiveNeuronDim), anti_aliasing=True)
      mag, ang = cv2.cartToPolar(dirX, -1*dirY, angleInDegrees=True)
      ang[mag == 0] = -100
      self.last_objects = deepcopy(self.objects)
      flow = np.zeros(shape=(self.dirSensitiveNeuronDim,self.dirSensitiveNeuronDim,2))
      flow[:,:,0] = dirX
      flow[:,:,1] = dirY
      goodInds = np.zeros(shape=(self.dirSensitiveNeuronDim,self.dirSensitiveNeuronDim))
    self.ldflow.append({'flow':flow,'mag':mag,'ang':ang,'goodInds':goodInds})

  def updateDirSensitiveRates (self):
    # update firing rate of dir sensitive neurons using dirs (2D array with motion direction at each coordinate)
    if len(self.ldflow) < 1: return
    dflow = self.ldflow[-1]
    motiondir = dflow['ang'] # angles in degrees, but thresholded for significant motion; negative value means not used
    dAngPeak = self.dAngPeak
    ds_courtXRng, ds_racket0XRng, ds_racketXRng = self.getNewCoords()
    if self.reducedNet:
      AngRFSigma2 = self.AngRFSigma2
      MaxRate = self.dirSensitiveNeuronRate[1]
      for pop in self.ldirpop: self.dFiringRates[pop] = self.dirSensitiveNeuronRate[0] * np.ones(shape=(1,1)) # should have a single angle per direction selective neuron pop
      court_motiondir = motiondir[:,ds_courtXRng[0]:ds_courtXRng[1]] # only motion direction of ball in the court
      unique_angles = np.unique(np.floor(court_motiondir))
      print('angles:',unique_angles)
      for a in unique_angles:
        if a >= 0.0:
          for pop in self.ldirpop:
            if self.EXPDir:
              fctr = np.exp(-1.0*(getangdiff(a,dAngPeak[pop])**2)/AngRFSigma2)
              if MaxRate * fctr < self.FiringRateCutoff: fctr = 0
              self.dFiringRates[pop][0] += MaxRate * fctr
            else:
              self.dFiringRates[pop][0] = MaxRate
      print(self.dFiringRates)
    else:
      dirSensitiveNeuronDim = self.dirSensitiveNeuronDim
      if motiondir.shape[0] != dirSensitiveNeuronDim or motiondir.shape[1] != dirSensitiveNeuronDim:
        motiondir = resize(motiondir, (dirSensitiveNeuronDim, dirSensitiveNeuronDim), anti_aliasing=True)
      AngRFSigma2 = self.AngRFSigma2
      MaxRate = self.dirSensitiveNeuronRate[1]
      for pop in self.ldirpop: self.dFiringRates[pop] = self.dirSensitiveNeuronRate[0] * np.ones(shape=(dirSensitiveNeuronDim,dirSensitiveNeuronDim))
      if self.EXPDir:
        for y in range(motiondir.shape[0]):
          for x in range(motiondir.shape[1]):
            if motiondir[y,x] >= 0.0: # make sure it's a valid angle
              for pop in self.ldirpop:
                fctr = np.exp(-1.0*(getangdiff(motiondir[y][x],dAngPeak[pop])**2)/AngRFSigma2)
                #print('updateDirRates',pop,x,y,fctr,dAngPeak[pop],motiondir[y][x])
                if MaxRate * fctr < self.FiringRateCutoff: fctr = 0
                self.dFiringRates[pop][y,x] += MaxRate * fctr
      else:
        for y in range(motiondir.shape[0]):
          for x in range(motiondir.shape[1]):
            if motiondir[y,x] >= 0.0: # make sure it's a valid angle
              for pop in self.ldirpop:
                if abs(getangdiff(motiondir[y][x],dAngPeak[pop])) <= self.AngRFSigma:
                  self.dFiringRates[pop][y,x] = MaxRate
                #print('updateDirRates',pop,x,y,fctr,dAngPeak[pop],motiondir[y][x])
      #print('motiondir',motiondir)
      for pop in self.ldirpop:
        self.dFiringRates[pop]=np.reshape(self.dFiringRates[pop],dirSensitiveNeuronDim**2)
        #print(pop,np.amin(self.dFiringRates[pop]),np.amax(self.dFiringRates[pop]),np.mean(self.dFiringRates[pop]))
        #print(pop,self.dFiringRates[pop])
    if dconf['sim']['saveAssignedFiringRates']:
      frcopy = deepcopy(self.dFiringRates)
      self.dAllFiringRates.append(frcopy)

  def updateDirSensitiveRatesWithPadding (self):
    # update firing rate of dir sensitive neurons using dirs (2D array with motion direction at each coordinate)
    if len(self.ldflow) < 1: return
    dflow = self.ldflow[-1]
    motiondir = dflow['ang'] # angles in degrees, but thresholded for significant motion; negative value means not used
    dAngPeak = self.dAngPeak
    dirSensitiveNeuronDim = self.dirSensitiveNeuronDim + self.dReceptiveField['EV1DE']-1
    offset = int((self.dReceptiveField['EV1DE']-1)/2)
    padded_motiondir = np.multiply(-100,np.ones(shape=(dirSensitiveNeuronDim,dirSensitiveNeuronDim)))
    padded_motiondir[offset:offset+motiondir.shape[0],offset:offset+motiondir.shape[1]]=motiondir
    AngRFSigma2 = self.AngRFSigma2
    MaxRate = self.dirSensitiveNeuronRate[1]
    for pop in self.ldirpop: self.dFiringRates[pop] = self.dirSensitiveNeuronRate[0] * np.ones(shape=(dirSensitiveNeuronDim,dirSensitiveNeuronDim))
    if self.EXPDir:
      for y in range(padded_motiondir.shape[0]):
        for x in range(padded_motiondir.shape[1]):
          if padded_motiondir[y,x] >= 0.0: # make sure it's a valid angle
            for pop in self.ldirpop:
              fctr = np.exp(-1.0*(getangdiff(padded_motiondir[y][x],dAngPeak[pop])**2)/AngRFSigma2)
              #print('updateDirRates',pop,x,y,fctr,dAngPeak[pop],padded_motiondir[y][x])
              if MaxRate * fctr < self.FiringRateCutoff: fctr = 0
              self.dFiringRates[pop][y,x] += MaxRate * fctr
    else:
      for y in range(padded_motiondir.shape[0]):
        for x in range(padded_motiondir.shape[1]):
          if padded_motiondir[y,x] >= 0.0: # make sure it's a valid angle
            for pop in self.ldirpop:
              if abs(getangdiff(padded_motiondir[y][x],dAngPeak[pop])) <= self.AngRFSigma:
                self.dFiringRates[pop][y,x] = MaxRate
              #print('updateDirRates',pop,x,y,fctr,dAngPeak[pop],padded_motiondir[y][x])
    #print('padded_motiondir',padded_motiondir)
    for pop in self.ldirpop:
      self.dFiringRates[pop]=np.reshape(self.dFiringRates[pop],dirSensitiveNeuronDim**2)
      #print(pop,np.amin(self.dFiringRates[pop]),np.amax(self.dFiringRates[pop]),np.mean(self.dFiringRates[pop]))
      #print(pop,self.dFiringRates[pop])
    if dconf['sim']['saveAssignedFiringRates']:
      frcopy = deepcopy(self.dFiringRates)
      self.dAllFiringRates.append(frcopy)
       
  def findobj (self, img, xrng, yrng):
    # find an object's x, y position in the image (assumes bright object on dark background)
    subimg = img[yrng[0]:yrng[1],xrng[0]:xrng[1],:]
    sIC = np.sum(subimg,2) #assuming the color of object is uniform, add values or r,g,b to get a single value      
    pixelVal = np.amax(sIC) #find the pixel value representing object assuming a black background
    sIC[sIC<pixelVal]=0 #make binary image
    Obj_inds = []
    for i in range(sIC.shape[0]):
      for j in range(sIC.shape[1]):
        if sIC[i,j]>0:
          Obj_inds.append([i,j])
    if sIC.shape[0]*sIC.shape[1]==np.shape(Obj_inds)[0]: #if there is no object in the subimage
      ypos = -1
      xpos = -1
    else:
      ypos = np.median(Obj_inds,0)[0] #y position of the center of mass of the object
      xpos = np.median(Obj_inds,0)[1] #x position of the center of mass of the object
    return xpos, ypos

  def predictBallRacketYIntercept(self, xpos1, ypos1, xpos2, ypos2):
    if ((xpos1==-1) or (xpos2==-1)):
      predY = -1
    else:
      deltax = xpos2-xpos1
      if deltax<=0:
        predY = -1
      else:
        if ypos1<0:
          predY = -1
        else:
          NB_intercept_steps = np.ceil((120.0 - xpos2)/deltax)
          deltay = ypos2-ypos1
          predY_nodeflection = ypos2 + (NB_intercept_steps*deltay)
          if predY_nodeflection<0:
            predY = -1*predY_nodeflection
          elif predY_nodeflection>160:
            predY = predY_nodeflection-160
          else:
            predY = predY_nodeflection
    return predY

  def getPaddedImage(self,gs_obs,padding_dim,courtXRng,racketXRng):
    expected_racket_len = 16
    #gs_obs = 255.0*rgb2gray(obs)
    input_dim = gs_obs.shape[0] + 2*padding_dim
    padded_Image = np.amin(gs_obs)*np.ones(shape=(input_dim,input_dim))
    padded_Image[padding_dim:padding_dim+gs_obs.shape[0],padding_dim:padding_dim+gs_obs.shape[1]] = gs_obs
    racket2 = gs_obs[:,courtXRng[0]-4:courtXRng[0]-1]
    if len(np.unique(racket2))>1:
      binary_racket2 = racket2 > self.getThreshold(racket2)
      racket2_ypixels = np.unique(np.where(binary_racket2)[0])
      racket2_len = len(racket2_ypixels)
    else:
      racket2_len =0
    racket1 = gs_obs[:,racketXRng[0]:racketXRng[1]]
    if len(np.unique(racket2))>1:
      binary_racket1 = racket1 > self.getThreshold(racket1)
      racket1_ypixels = np.unique(np.where(binary_racket1)[0])
      racket1_len = len(racket1_ypixels)
    else:
      racket1_len = 0
    if racket1_len>0 and expected_racket_len>racket1_len:
      missing_racket1_len = expected_racket_len-racket1_len
      if 0 in racket1_ypixels:
        for ind in range(missing_racket1_len):
          for jind in range(racketXRng[1]-racketXRng[0]+1):
            padded_Image[padding_dim-ind,racketXRng[0]+padding_dim+jind-1] = racket1[0,0]
      else:
        for ind in range(missing_racket1_len):
          for jind in range(racketXRng[1]-racketXRng[0]+1):
            padded_Image[gs_obs.shape[0]+padding_dim+ind,racketXRng[0]+padding_dim+jind-1] = racket1[-1,0]
    if racket2_len>0 and expected_racket_len>racket2_len:
      missing_racket2_len = expected_racket_len-racket2_len
      if 0 in racket2_ypixels:
        for ind in range(missing_racket2_len):
          for jind in range(racketXRng[1]-racketXRng[0]+1):
            padded_Image[padding_dim-1-ind,courtXRng[0]+padding_dim-1-jind] = racket2[0,0]
      else:
        for ind in range(missing_racket2_len):
          for jind in range(racketXRng[1]-racketXRng[0]+1):
            padded_Image[gs_obs.shape[0]+padding_dim+ind,courtXRng[0]+padding_dim-1-jind] = racket2[-1,0]
    return padded_Image

  def downscale (self, I, downshape):
    #x = resize(I, (I.shape[0]/self.downsampshape[0],I.shape[1]/self.downsampshape[1]), anti_aliasing=False)#,anti_aliasing_sigma=2)
    #return x
    #erosion = cv2.erode(x,np.ones((2,2),np.uint8),iterations = 1)
    #return erosion
    #return cv2.resize( cv2.resize(I,(0,0),fx=4,fy=4,interpolation=cv2.INTER_LINEAR), (0,0), fx=0.25/downshape[0], fy=0.25/downshape[1], interpolation = cv2.INTER_NEAREST)
    #return cv2.resize(I, (int(I.shape[0]/downshape[0]),int(I.shape[1]/downshape[1])), interpolation = cv2.INTER_NEAREST
    if downshape[0] >= 4:
      return downscale_local_mean(I,downshape)
    else:
      return I[::downshape[0],::downshape[1]] # downsample by factor of shape      
    #x = resize(I, (I.shape[0]*8,I.shape[1]*8))
    #return resize(x, (I.shape[0]/(8*self.downsampshape[0]),I.shape[1]/(8*self.downsampshape[1])))    
    #return resize(I, (I.shape[0]/self.downsampshape[0],I.shape[1]/self.downsampshape[1]), anti_aliasing=True,anti_aliasing_sigma=1.5)
    #return I.astype(np.float).ravel()
  def avoidStuckRule(self):
    ypos_Racket = self.dObjPos['ball'][-1][1]
    if ypos_Racket <= 8:
      print('STUCK MOVE DOWN, YPOS RACKET=',ypos_Racket)
      caction = dconf['moves']['DOWN']
    elif ypos_Racket >= 152:
      print('STUCK MOVE UP, YPOS RACKET=',ypos_Racket)
      caction = dconf['moves']['UP']                      
    elif ypos_Racket - 1 - self.racketH*0.8125 <= 0 and caction==dconf['moves']['UP']:
      print('STUCK STOP UP, YPOS RACKET=', ypos_Racket, 'bound=',ypos_Racket - 1 - self.racketH/2)
      caction = dconf['moves']['NOMOVE']
    elif ypos_Racket + 1 + self.racketH*0.8125 >= self.maxYPixel and caction==dconf['moves']['DOWN']:
      print('STUCK STOP DOWN, YPOS RACKET=',ypos_Racket, 'bound=',ypos_Racket + 1 + self.racketH/2)
      caction = dconf['moves']['NOMOVE']
    return caction    

  def followTheBallRule(self,simtime):
    if np.shape(self.last_obs)[0]>0: #if last_obs is not empty              
      xpos_Ball, ypos_Ball = self.findobj(self.last_obs, self.courtXRng, self.courtYRng) # get x,y positions of ball
      xpos_Racket, ypos_Racket = self.findobj(self.last_obs, self.racketXRng, self.courtYRng) # get x,y positions of racket
      #Now we know the position of racket relative to the ball. We can suggest the action for the racket so that it doesn't miss the ball.
      #For the time being, I am implementing a simple rule i.e. based on only the ypos of racket relative to the ball
      if ypos_Ball==-1: #guess about proposed move can't be made because ball was not visible in the court
        proposed_action = -1 #no valid action guessed
      elif ypos_Racket>ypos_Ball: #if the racket is lower than the ball the suggestion is to move up
        proposed_action = dconf['moves']['UP'] #move up
      elif ypos_Racket<ypos_Ball: #if the racket is higher than the ball the suggestion is to move down
        proposed_action = dconf['moves']['DOWN'] #move down
      elif ypos_Racket==ypos_Ball:
        proposed_action = dconf['moves']['NOMOVE'] #no move
      #self.FullImages.append(np.sum(self.last_obs[courtYRng[0]:courtYRng[1],:,:],2))
      if xpos_Ball>0 and ypos_Ball>0:
        self.dObjPos['ball'].append([self.courtXRng[0]+xpos_Ball,ypos_Ball])
      else:
        self.dObjPos['ball'].append([-1,-1])
      if xpos_Racket>0 and ypos_Racket>0:
        self.dObjPos['racket'].append([self.racketXRng[0]+xpos_Racket,ypos_Racket])
      else:
        self.dObjPos['racket'].append([-1,-1])
      self.dObjPos['time'].append(simtime)
    else:
      proposed_action = -1 #if there is no last_obs
      ypos_Ball = -1 #if there is no last_obs, no position of ball
      xpos_Ball = -1 #if there is no last_obs, no position of ball
      self.dObjPos['ball'].append([-1,-1])
      self.dObjPos['racket'].append([-1,-1])
    return proposed_action

  def useRacketPredictedPos(self, observation, proposed_action):
    FollowTargetSign = 0
    xpos_Ball = self.dObjPos['ball'][-1][0]-self.courtXRng[0]
    ypos_Ball = self.dObjPos['ball'][-1][1]
    xpos_Racket = self.dObjPos['racket'][-1][0]-self.courtXRng[0]
    ypos_Racket = self.dObjPos['racket'][-1][1]
    xpos_Ball2, ypos_Ball2 = self.findobj(observation, self.courtXRng, self.courtYRng)
    ball_moves_towards_racket = False
    if xpos_Ball>0 and xpos_Ball2>0:
      if xpos_Ball2-xpos_Ball>0:
        ball_moves_towards_racket = True # use proposed action for reward only when the ball moves towards the racket
        current_ball_dir = 1 
      elif xpos_Ball2-xpos_Ball<0:
        ball_moves_towards_racket = False
        current_ball_dir = -1
      else:
        ball_moves_towards_racket = False
        current_ball_dir = 0 #direction can't be determinted  prob. because the ball didn't move in x dir.
    else:
      ball_moves_towards_racket = False
      current_ball_dir = 0 #direction can't be determined because either current or last position of the ball is outside the court
    skipPred = False # skip prediction of y intercept?
    if dconf["followOnlyTowards"] and not ball_moves_towards_racket:
      proposed_action = -1 # no proposed action if ball moving away from racket
      skipPred = True # skip prediction if ba
    if not skipPred and dconf["useRacketPredictedPos"]:
      xpos_Racket2, ypos_Racket2 = self.findobj (observation, self.racketXRng, self.courtYRng)
      predY = self.predictBallRacketYIntercept(xpos_Ball,ypos_Ball,xpos_Ball2,ypos_Ball2)
      if predY==-1:
        proposed_action = -1
      else:
        targetY = ypos_Racket2 - predY
        if targetY>8:
          proposed_action = dconf['moves']['UP'] #move up
        elif targetY<-8:
          proposed_action = dconf['moves']['DOWN'] #move down
        else:
          proposed_action = dconf['moves']['NOMOVE'] #no move
        YDist = abs(ypos_Racket - predY) # pre-move distance to predicted y intercept
        YDist2 = abs(ypos_Racket2 - predY) # post-move distance to predicted y intercept
        if YDist2 < YDist: # smaller distance to target? set to positive value (reward)
          FollowTargetSign = 1
        elif YDist2 > YDist: # larger distance to target? set to negative value (punishment)
          FollowTargetSign = -1
    return proposed_action, FollowTargetSign, current_ball_dir, xpos_Ball2

  def playGame (self, actions, epCount, simtime): #actions need to be generated from motor cortex
    # PLAY GAME
    rewards = []; proposed_actions =[]; total_hits = []; Images = []; FollowTargetSign = 0
    input_dim = self.input_dim
    done = False
    courtYRng, courtXRng, racketXRng = self.courtYRng, self.courtXRng, self.racketXRng # coordinate ranges for different objects (PONG-specific)    
    if self.intaction==1:
      lgwght = [1.0]
    else:
      lgwght = np.linspace(0.6, 1, self.intaction) # time-decay grayscale image weights (earlier indices with lower weights are from older frames)
    lgimage = [] # grayscale down-sampled images with decaying time-lagged input
    lgimage_ns = [] #grayscale full images with decaying time-lagged input 
    if len(self.last_obs)==0: #if its the first action of the episode, there won't be any last_obs, therefore no last image
      lobs_gimage_ds = []
    else:
      lobs_gimage = 255.0*rgb2gray(self.last_obs[courtYRng[0]:courtYRng[1],:,:])
      if self.useImagePadding: 
        padding_dim = self.padPixelEachSide
        lobs_gimage = self.getPaddedImage(lobs_gimage,padding_dim,courtXRng,racketXRng)
      #lobs_gimage_ds = self.downscale(lobs_gimage, self.downsampshape)
      lobs_gimage_ds = self.downscale(lobs_gimage,self.downsampshape)
      lobs_gimage_ds = np.where(lobs_gimage_ds>np.min(lobs_gimage_ds)+1,255,lobs_gimage_ds)
      lobs_gimage_ds = 0.5*lobs_gimage_ds #use this image for motion computation only
      
    for adx in range(self.intaction):
      #for each action generated by the firing rate of the motor cortex, find the suggested-action by comparing the position of the ball and racket 
      caction = actions[adx] #action generated by the firing rate of the motor cortex
      proposed_action = self.followTheBallRule(simtime)
      # do not allow racket to get stuck at top or bottom
      if self.avoidStuck and np.shape(self.last_obs)[0]>0:
        caction = self.avoidStuckRule()   # if the ball is stuck, this will bypass the action generated by the motor cortex
      if useSimulatedEnv:
        observation, reward, done = self.pong.step(caction)
      else:
        observation, reward, done, info = self.env.step(caction) # Re-Initializes reward before if statement
        # To eliminate momentum
        # print('Here is caction: ' , caction)
        if caction in [dconf['moves']['DOWN'], dconf['moves']['UP'], dconf['moves']['NOMOVE']]:
          # Follow down/up/stay with stay to prevent momentum problem (Pong-specific)
          stay_step = 0 # initialize
          while not done and stay_step < self.stayStepLim:
            observation, interreward, done, info = env.step(dconf['moves']['NOMOVE']) # Stay motion
            reward += interreward  # Uses summation so no reinforcement/punishment is missed
            stay_step += 1
          env.render() # Renders the game after the stay steps
      #find position of ball after action
      proposed_action, FollowTargetSign, current_ball_dir, xpos_Ball2 = self.useRacketPredictedPos(observation, proposed_action)
      ball_hits_racket = 0
      # previously I assumed when current_ball_dir is 0 there is no way to find out if the ball hit the racket
      if current_ball_dir-self.last_ball_dir<0 and reward==0 and xpos_Ball2>courtXRng[1]-courtXRng[0]-40:
        ball_hits_racket = 1
      #print('Current_ball_dir',current_ball_dir,'Last ball dir',self.last_ball_dir,'current X pos Ball', xpos_Ball2,'last X pos Ball', xpos_Ball)
      #print(ball_hits_racket)
      self.last_ball_dir = current_ball_dir
      total_hits.append(ball_hits_racket) # i dont think this can be more than a single hit in 5 moves. so check if sum is greater than 1, print error
      if not useSimulatedEnv:
        self.env.render()
      self.last_obs = observation # current observation will be used as last_obs for the next action
      if done:
        if not useSimulatedEnv: self.env.reset()
        self.last_obs = [] # when the game ends, and new game starts, there is no last observation
        self.last_ball_dir=0
        done = False
      rewards.append(reward)
      proposed_actions.append(proposed_action)    
      gray_Image = 255.0*rgb2gray(observation[courtYRng[0]:courtYRng[1],:,:]) # convert to grayscale; rgb2gray has 0-1 range so mul by 255
      if self.useImagePadding: 
        padding_dim = self.padPixelEachSide
        gray_Image = self.getPaddedImage(gray_Image,padding_dim,courtXRng,racketXRng)
      gray_ds = self.downscale(gray_Image,self.downsampshape) # then downsample
      gray_ds = np.where(gray_ds>np.min(gray_ds)+1,255,gray_ds) # Different thresholding
      gray_ns = np.where(gray_Image>np.min(gray_Image)+1,255,gray_Image)
      lgimage_ns.append(lgwght[adx]*gray_ns)
      lgimage.append(lgwght[adx]*gray_ds) # save weighted grayscale image from current frame
      self.countAll += 1

    # NB: previously we merged 2x2 pixels into 1 value. Now we merge 8x8 pixels into 1 value.
    # so the original 160x160 pixels will result into 20x20 values instead of previously used 80x80.        
    if len(lgimage)>1:
      dsum_Images = np.maximum(lgimage[0],lgimage[1])
      nsum_Images = np.maximum(lgimage_ns[0],lgimage_ns[1])
      for gimage in lgimage[2:]: dsum_Images = np.maximum(dsum_Images,gimage)
      for gimage in lgimage_ns[2:]: nsum_Images = np.maximum(nsum_Images,gimage)
    else:
      dsum_Images = lgimage[0]
      nsum_Images = lgimage_ns[0]
    self.FullImages.append(nsum_Images) # save full images ----> THIS IS JUST USED FOR DIRECTIONS (for accuracy)
    if self.binary_Image is not None:
      self.ReducedImages.append(255.0 * self.binary_Image)
    else:
      self.ReducedImages.append(dsum_Images) # save the input image
    if dconf['net']['useNeuronPad']==1:
      self.updateInputRatesWithPadding(dsum_Images)
    else:
      self.updateInputRates(dsum_Images) # update input rates to retinal neurons
    if self.intaction==1: #if only one frame used per play, then add the downsampled and scaled image from last_obs for direction computation 
      if len(lobs_gimage_ds)>0:
        dsum_Images = np.maximum(dsum_Images,lobs_gimage_ds)

    if self.dirSensitiveNeuronDim > 0: # as long as we have direction selective neurons
      if dconf['DirectionDetectionAlgo']['OpticFlow']:
        self.computeMotionFields(UseFull=dconf['DirectionDetectionAlgo']['UseFull']) # compute the motion fields
      elif dconf['DirectionDetectionAlgo']['CentroidTracker']:
        # compute the motion field using CetroidTracking
        self.computeAllObjectsMotionDirections(UseFull=dconf['DirectionDetectionAlgo']['UseFull']) 
      if dconf['net']['useNeuronPad']==1:
        self.updateDirSensitiveRatesWithPadding()
      else:
        self.updateDirSensitiveRates() # update motion sensitive neuron input rates

    if done: # done means that 1 episode of the game finished, so the environment needs to be reset. 
      epCount.append(self.countAll)
      if not useSimulatedEnv:
        self.env.reset()
        #self.env.frameskip = 3 # why is frameskip set to 3??
      self.countAll = 0 
    if np.sum(total_hits)>1:
      print('ERROR COMPUTING NUMBER OF HITS')
    for r in range(len(rewards)):
      if rewards[r]==-1: total_hits[r]=-1 #when the ball misses the racket, the reward is -1
    return rewards, epCount, proposed_actions, total_hits, FollowTargetSign
            
