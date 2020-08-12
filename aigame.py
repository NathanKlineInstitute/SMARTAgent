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
    self.dReceptiveField = OrderedDict({pop:np.amax(list(dconf['net']['alltopolconvcons'][pop].values())) for pop in self.lratepop})
    self.dInputs = OrderedDict({pop:int((np.sqrt(dconf['net']['allpops'][pop])+self.dReceptiveField[pop]-1)**2) for pop in self.lratepop})
    for d in self.ldir: self.lratepop.append('EV1D'+d)
    self.dFVec = OrderedDict({pop:h.Vector() for pop in self.lratepop}) # NEURON Vectors for firing rate calculations
    #if dconf['net']['useNeuronPad']==1:
    #  self.dFiringRates = OrderedDict({pop:np.zeros(self.dInputs[pop]) for pop in self.lratepop}) # python objects for firing rate calculations
    #else:
    self.dFiringRates = OrderedDict({pop:np.zeros(dconf['net']['allpops'][pop]) for pop in self.lratepop}) # python objects for firing rate calculations
    if dconf['net']['useNeuronPad']==1:
      self.dFiringRates[self.InputPop] = np.zeros(self.dInputs[self.InputPop])
    self.dAngPeak = OrderedDict({'EV1DE': 0.0,'EV1DNE': 45.0, # receptive field peak angles for the direction selective populations
                                'EV1DN': 90.0,'EV1DNW': 135.0,
                                'EV1DW': 180.0,'EV1DSW': 235.0,
                                'EV1DS': 270.0,'EV1DSE': 315.0})
    self.AngRFSigma = dconf['net']['AngRFSigma']
    self.AngRFSigma2 = dconf['net']['AngRFSigma']**2 # angular receptive field (RF) sigma squared used for dir selective neuron RFs
    if self.AngRFSigma2 <= 0.0: self.AngRFSigma2=1.0
    self.EXPDir = True
    if 'EXPDir' in dconf['net']: self.EXPDir = dconf['net']['EXPDir']
    if dconf['net']['useNeuronPad']==1:
      self.input_dim = int(np.sqrt(self.dInputs[self.InputPop]))
    else:  
      self.input_dim = int(np.sqrt(dconf['net']['allpops'][self.InputPop])) # input image XY plane width,height -- not used anywhere    
    self.dirSensitiveNeuronDim = int(np.sqrt(dconf['net']['allpops']['EV1DE'])) # direction sensitive neuron XY plane width,height
    self.dirSensitiveNeuronRate = (dconf['net']['DirMinRate'], dconf['net']['DirMaxRate']) # min, max firing rate (Hz) for dir sensitive neurons
    self.intaction = int(dconf['actionsPerPlay']) # integrate this many actions together before returning reward information to model
    # these are Pong-specific coordinate ranges; should later move out of this function into Pong-specific functions
    self.courtYRng = (34, 194) # court y range
    self.courtXRng = (20, 140) # court x range
    self.racketXRng = (141, 144) # racket x range
    self.dObjPos = {'racket':[], 'ball':[]}
    self.last_obs = [] # previous observation
    self.last_ball_dir = 0 # last ball direction
    self.FullImages = [] # full resolution images from game environment
    self.ReducedImages = [] # low resolution images from game environment used as input to neuronal network model
    self.ldflow = [] # list of dictionary of optical flow (motion) fields
    if dconf['DirectionDetectionAlgo']['CentroidTracker']:
      self.ct = CentroidTracker()
      self.objects = OrderedDict() # objects detected in current frame
      self.last_objects = OrderedDict() # objects detected in previous frame
    if "stayStepLim" in dconf:
      self.stayStepLim = dconf['stayStepLim']
    else:
      self.stayStepLim = 6 # number of steps to hold still after every move (to reduce momentum)
      # Takes 6 stays instead of 3 because it seems every other input is ignored (check dad_notes.txt for details)
    self.downsampshape = (8,8) # default is 20x20 (20 = 1/8 of 160)
    if dconf['net']['allpops'][self.InputPop] == 1600: # this is for 40x40 (40 = 1/4 of 160)
      self.downsampshape = (4,4)
    elif dconf['net']['allpops'][self.InputPop] == 6400: # this is for 80x80 (1/2 resolution)
      self.downsampshape = (2,2)
    elif dconf['net']['allpops'][self.InputPop] == 25600: # this is for 160x160 (full resolution)
      self.downsampshape = (1,1)

  def updateInputRatesWithPadding (self, dsum_Images):
    # update input rates to retinal neurons
    padded_Image = np.amin(dsum_Images)*np.ones(shape=(self.input_dim,self.input_dim))
    offset = int((self.dReceptiveField[self.InputPop]-1)/2)
    padded_Image[offset:offset+dsum_Images.shape[0],offset:offset+dsum_Images.shape[1]]=dsum_Images
    if dconf['net']['useBinaryImage']==1:
      thresh = threshold_otsu(padded_Image)
      binary_Image = padded_Image > thresh
      fr_Images = 50.0*binary_Image
    else:
      padded_Image = padded_Image - np.amin(padded_Image)
      padded_Image = (255.0/np.amax(padded_Image))*padded_Image # this will make sure that padded_Image spans 0-255
      fr_Images = 50/(1+np.exp((np.multiply(-1,padded_Image)+123)/10))
      #fr_Images = np.subtract(fr_Images,np.min(fr_Images)) #baseline firing rate subtraction. Instead all excitatory neurons are firing at 5Hz.
      #print(np.amin(fr_Images),np.amax(fr_Images))
    self.dFiringRates[self.InputPop] = np.reshape(fr_Images,self.dInputs[self.InputPop]) #400 for 20*20, 900 for 30*30, etc.

  def updateInputRates (self, dsum_Images):
    # update input rates to retinal neurons
    #fr_Images = np.where(dsum_Images>1.0,100,dsum_Images) #Using this to check what number would work for firing rate
    #fr_Images = np.where(dsum_Images<10.0,0,dsum_Images)
    if dconf['net']['useBinaryImage']==1:
      thresh = threshold_otsu(dsum_Images)
      binary_Image = dsum_Images > thresh
      fr_Images = 50.0*binary_Image
    else:
      dsum_Images = dsum_Images - np.amin(dsum_Images)
      dsum_Images = (255.0/np.amax(dsum_Images))*dsum_Images
      fr_Images = 50/(1+np.exp((np.multiply(-1,dsum_Images)+123)/10))
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
      self.ldflow.append({'flow':flow,'mag':mag,'ang':ang,'goodInds':goodInds,'thang':ang,'thflow':flow})
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
      mag, ang = cv2.cartToPolar(dirX, -1*dirY)
      ang = np.rad2deg(ang)
      ang[mag == 0] = -100
      self.last_objects = deepcopy(self.objects)
      flow = np.zeros(shape=(self.dirSensitiveNeuronDim,self.dirSensitiveNeuronDim,2))
      flow[:,:,0] = dirX
      flow[:,:,1] = dirY
      goodInds = np.zeros(shape=(self.dirSensitiveNeuronDim,self.dirSensitiveNeuronDim))
    self.ldflow.append({'flow':flow,'mag':mag,'ang':ang,'goodInds':goodInds,'thang':ang,'thflow':flow})

  def updateDirSensitiveRates (self):
    # update firing rate of dir sensitive neurons using dirs (2D array with motion direction at each coordinate)
    if len(self.ldflow) < 1: return
    dflow = self.ldflow[-1]
    motiondir = dflow['thang'] # angles in degrees, but thresholded for significant motion; negative value means not used
    dAngPeak = self.dAngPeak
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

  def playGame (self, actions, epCount): #actions need to be generated from motor cortex
    # PLAY GAME
    rewards = []; proposed_actions =[]; total_hits = []; Images = []
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
      lobs_gimage_ds = downscale_local_mean(lobs_gimage,self.downsampshape)
      lobs_gimage_ds = np.where(lobs_gimage_ds>np.min(lobs_gimage_ds)+1,255,lobs_gimage_ds)
      lobs_gimage_ds = 0.5*lobs_gimage_ds #use this image for motion computation only
      
    for adx in range(self.intaction):
      #for each action generated by the firing rate of the motor cortex, find the suggested-action by comparing the position of the ball and racket 
      caction = actions[adx] #action generated by the firing rate of the motor cortex

      if np.shape(self.last_obs)[0]>0: #if last_obs is not empty              
        xpos_Ball, ypos_Ball = self.findobj(self.last_obs, courtXRng, courtYRng) # get x,y positions of ball
        xpos_Racket, ypos_Racket = self.findobj(self.last_obs, racketXRng, courtYRng) # get x,y positions of racket          
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
        self.dObjPos['ball'].append([courtXRng[0]-1+xpos_Ball,ypos_Ball])
        self.dObjPos['racket'].append([racketXRng[0]-1+xpos_Racket,ypos_Racket])
      else:
        proposed_action = -1 #if there is no last_obs
        ypos_Ball = -1 #if there is no last_obs, no position of ball
        xpos_Ball = -1 #if there is no last_obs, no position of ball
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
      xpos_Ball2, ypos_Ball2 = self.findobj(observation, courtXRng, courtYRng)
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
      if "followOnlyTowards" in dconf:
        if dconf["followOnlyTowards"] and not ball_moves_towards_racket:
          proposed_action = -1 # no proposed action if ball moving away from racket
          skipPred = True # skip prediction if ba
      
      if not skipPred and "useRacketPredictedPos" in dconf:
        if dconf["useRacketPredictedPos"]:
          xpos_Racket2, ypos_Racket2 = self.findobj (observation, racketXRng, courtYRng)
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

      ball_hits_racket = 0
      # previously I assumed when current_ball_dir is 0 there is no way to find out if the ball hit the racket
      if current_ball_dir-self.last_ball_dir<0 and reward==0 and xpos_Ball2>courtXRng[1]-courtXRng[0]-40:
        ball_hits_racket = 1
      #print('Current_ball_dir', current_ball_dir)
      #print('Last ball dir', self.last_ball_dir)
      #print('current X pos Ball', xpos_Ball2)
      #print('last X pos Ball', xpos_Ball)
      #print('Court Range',courtXRng) 
      print(ball_hits_racket)
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
      gray_ds = downscale_local_mean(gray_Image,self.downsampshape) # then downsample
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
    self.ReducedImages.append(dsum_Images) # save the input image
    if dconf['net']['useNeuronPad']==1:
      self.updateInputRatesWithPadding(dsum_Images)
    else:
      self.updateInputRates(dsum_Images) # update input rates to retinal neurons
    if self.intaction==1: #if only one frame used per play, then add the downsampled and scaled image from last_obs for direction computation 
      if len(lobs_gimage_ds)>0:
        dsum_Images = np.maximum(dsum_Images,lobs_gimage_ds)
    if dconf['DirectionDetectionAlgo']['OpticFlow']:
      self.computeMotionFields(UseFull=dconf['DirectionDetectionAlgo']['UseFull']) # compute the motion fields
    elif dconf['DirectionDetectionAlgo']['CentroidTracker']:
      self.computeAllObjectsMotionDirections(UseFull=dconf['DirectionDetectionAlgo']['UseFull']) # compute the motion field using CetroidTracking
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
    return rewards, epCount, proposed_actions, total_hits
            
