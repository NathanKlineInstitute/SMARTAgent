"""
AIGame: connects OpenAI gym game to the model (V1-M1-RL)
Adapted from arm.py
Original Version: 2015jan28 by salvadordura@gmail.com
Modified Version: 2019oct1 by haroon.anwar@gmail.com
Modified 2019-2020 samn
"""

from neuron import h
from pylab import concatenate, figure, show, ion, ioff, pause,xlabel, ylabel, plot, Circle, sqrt, arctan, arctan2, close
from copy import copy
from random import uniform, seed, sample, randint
from matplotlib import pyplot as plt
import random
import numpy as np
from skimage.transform import downscale_local_mean
from skimage.color import rgb2gray
import json
import gym
import sys
from gym import wrappers
from time import time
from collections import OrderedDict

# make the environment - env is global so that it only gets created on a single node (important when using MPI with > 1 node)
try:
  from conf import dconf
  env = gym.make(dconf['env']['name'],frameskip=dconf['env']['frameskip'])
  if dconf['env']['savemp4']: env = wrappers.Monitor(env, './videos/' + dconf['sim']['name'] + '/',force=True)
  env.reset()
except:
  print('Exception in makeENV')
  env = gym.make('Pong-v0',frameskip=3)
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
    self.env = env
    self.countAll = 0
    self.ldir = ['E','NE','N', 'NW','W','SW','S','SE']
    self.ldirpop = ['EV1D'+Dir for Dir in self.ldir]
    self.lratepop = ['ER'] # populations that we calculate rates for
    for d in self.ldir: self.lratepop.append('EV1D'+d)
    self.dFVec = OrderedDict({pop:h.Vector() for pop in self.lratepop}) # NEURON Vectors for firing rate calculations
    self.dFiringRates = OrderedDict({pop:np.zeros(dconf['net'][pop]) for pop in self.lratepop}) # python objects for firing rate calculations
    self.dAngPeak = OrderedDict({'EV1DE': 0.0,'EV1DNE': 45.0, # receptive field peak angles for the direction selective populations
                                'EV1DN': 90.0,'EV1DNW': 135.0,
                                'EV1DW': 180.0,'EV1DSW': 235.0,
                                'EV1DS': 270.0,'EV1DSE': 315.0})
    self.AngRFSigma2 = dconf['net']['AngRFSigma']**2 # angular receptive field (RF) sigma squared used for dir selective neuron RFs
    self.input_dim = int(np.sqrt(dconf['net']['ER'])) # input image XY plane width,height
    self.dirSensitiveNeuronDim = int(np.sqrt(dconf['net']['EV1DE'])) # direction sensitive neuron XY plane width,height
    self.dirSensitiveNeuronRate = (dconf['net']['DirMinRate'], dconf['net']['DirMaxRate']) # min, max firing rate (Hz) for dir sensitive neurons
    self.intaction = int(dconf['actionsPerPlay']) # integrate this many actions together before returning reward information to model
    # these are Pong-specific coordinate ranges; should later move out of this function into Pong-specific functions
    self.courtYRng = (34, 194)
    self.courtXRng = (20, 140)
    self.racketXRng = (141, 144)
    self.last_obs = []
    self.last_ball_dir = 0

  def updateInputRates (self, dsum_Images):
    # update input rates to retinal neurons
    #fr_Images = np.where(dsum_Images>1.0,100,dsum_Images) #Using this to check what number would work for firing rate
    #fr_Images = np.where(dsum_Images<10.0,0,dsum_Images)
    fr_Images = 40/(1+np.exp((np.multiply(-1,dsum_Images)+123)/25))
    fr_Images = np.subtract(fr_Images,np.min(fr_Images)) #baseline firing rate subtraction. Instead all excitatory neurons are firing at 5Hz.
    #print(np.amax(fr_Images))
    self.dFiringRates['ER'] = np.reshape(fr_Images,400) #400 for 20*20

  def computeMotion (self, dsum_Images):
    #compute directions of motion for every other pixel.
    bkgPixel = np.min(dsum_Images) # background pixel value
    dirSensitiveNeuronDim = self.dirSensitiveNeuronDim
    dirSensitiveNeurons = np.zeros(shape=(dirSensitiveNeuronDim,dirSensitiveNeuronDim))
    for dSNeuronX in range(dirSensitiveNeuronDim):
      Rx = 2*dSNeuronX
      if Rx==0:
        Rxs = [Rx,Rx+1,Rx+2]
      elif Rx==1:
        Rxs = [Rx-1, Rx, Rx+1, Rx+2]
      #elif Rx==dirSensitiveNeuronDim-1:
      #    Rxs = [Rx-2,Rx-1,Rx]
      elif Rx==((2*dirSensitiveNeuronDim)-2):
        Rxs = [Rx-2,Rx-1,Rx,Rx+1]
      else:
        Rxs = [Rx-2,Rx-1,Rx,Rx+1,Rx+2]
      for dSNeuronY in range(dirSensitiveNeuronDim):
        Ry = 2*dSNeuronY
        #print('Ry:',Ry)
        if Ry==0:
          Rys = [Ry, Ry+1, Ry+2]
        elif Ry==1:
          Rys = [Ry-1, Ry, Ry+1, Ry+2]
        #elif Ry==dirSensitiveNeuronDim-1:
        #    Rys = [Ry-2,Ry-1,Ry]
        elif Ry==((2*dirSensitiveNeuronDim)-2):
          Rys = [Ry-2,Ry-1,Ry,Ry+1]
        else:
          Rys = [Ry-2,Ry-1,Ry,Ry+1,Ry+2]
        #print('Xinds',Rxs,'Yinds',Rys)
        FOV = np.zeros(shape=(len(Rxs),len(Rys))) # field of view
        for xinds in range(len(Rxs)):
          for yinds in range(len(Rys)):
            FOV[xinds,yinds] = dsum_Images[Rxs[xinds],Rys[yinds]]
        #print(FOV)
        max_value = np.amax(FOV)
        max_ind = np.where(FOV==max_value)
        #print('max inds', max_ind) 
        #since the most recent frame has highest pixel intensity, any pixel with the maximum intensity will be
        #most probably the final instance of the object motion in that field of view
        bkg_inds = np.where(FOV == bkgPixel)
        if len(bkg_inds[0])>0:
          for yinds in range(len(bkg_inds[0])):
            ix = bkg_inds[0][yinds]
            iy = bkg_inds[1][yinds]
            FOV[ix,iy] = 1000
        #I dont want to compute object motion vector relative to the background. so to ignore background pixels, replacing them with large value
        #np.put(FOV,bkg_inds,1000) 
        min_value = np.amin(FOV)
        min_ind = np.where(FOV==min_value)
        #print('min inds', min_ind)
        #since latest frame has lowest pixel intensity (after ignoring background), any pixel with max intensity will
        #most probably be first instance of object motion in that field of view
        if len(max_ind[0])>len(min_ind[0]):
          mL = len(min_ind[0])
        elif len(max_ind[0])<len(min_ind[0]):
          mL = len(max_ind[0])
        else:
          mL = len(max_ind[0])
        #direction of the object motion in a field of view over last 5 frames/observations.
        dir1 = [max_ind[0][range(mL)]-min_ind[0][range(mL)],max_ind[1][range(mL)]-min_ind[1][range(mL)]] 
        dir2 = [np.median(dir1[1]),-1*np.median(dir1[0])] #flip y because indexing starts from top left.
        dirMain = [1,0] #using a reference for 0 degrees....considering first is for rows and second is for columns
        ndir2 = dir2 / np.linalg.norm(dir2)
        ndirMain = dirMain / np.linalg.norm(dirMain)
        theta = np.degrees(np.arccos(np.dot(ndir2,ndirMain))) #if theta is nan, no movement is detected
        if dir2[1]<0: theta = 360-theta 
        dirSensitiveNeurons[dSNeuronX,dSNeuronY] = theta # the motion angle (theta) at position dSNeuronX,dSNeuronY is stored
        #if not np.isnan(theta): print('Theta for FOV ',FOV,' is: ', theta)
    print('Computed angles:', dirSensitiveNeurons)
    return dirSensitiveNeurons
      
  def updateDirSensitiveRates (self, motiondir):
    # update firing rate of dir sensitive neurons using dirs (2D array with motion direction at each coordinate)
    dAngPeak = self.dAngPeak
    dirSensitiveNeuronDim = self.dirSensitiveNeuronDim
    AngRFSigma2 = self.AngRFSigma2
    MaxRate = self.dirSensitiveNeuronRate[1]
    for pop in self.ldirpop: self.dFiringRates[pop] = self.dirSensitiveNeuronRate[0] * np.ones(shape=(dirSensitiveNeuronDim,dirSensitiveNeuronDim))
    for y in range(motiondir.shape[0]):
      for x in range(motiondir.shape[1]):
        theta = motiondir[y][x]
        if np.isnan(theta): continue # skip invalid angles
        for pop in self.ldirpop:
          fctr = np.exp(-1.0*(getangdiff(theta,dAngPeak[pop])**2)/AngRFSigma2)
          # print('updateDirSensitiveRates',pop,x,y,fctr,dAngPeak[pop],motiondir[y][x])
          if fctr > 0.:
            self.dFiringRates[pop][y,x] += MaxRate * fctr
    for pop in self.ldirpop: self.dFiringRates[pop]=np.reshape(self.dFiringRates[pop],dirSensitiveNeuronDim**2) 
          
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

  def playGame (self, actions, epCount, InputImages): #actions need to be generated from motor cortex
    # PLAY GAME
    rewards = []; proposed_actions =[]; total_hits = []; Images = []; Ball_pos = []; Racket_pos = []
    input_dim = self.input_dim
    done = False
    courtYRng, courtXRng, racketXRng = self.courtYRng, self.courtXRng, self.racketXRng # coordinate ranges for different objects (PONG-specific)      
    lgwght = np.linspace(0.6, 1, self.intaction) # time-decay grayscale image weights (earlier indices with lower weights are from older frames)
    lgimage = [] # grayscale images with decaying time-lagged input

    for adx in range(self.intaction):
      #for each action generated by the firing rate of the motor cortex, find the suggested-action by comparing the position of the ball and racket 
      caction = actions[adx] #action generated by the firing rate of the motor cortex

      if np.shape(self.last_obs)[0]>0: #if last_obs is not empty              
        xpos_Ball, ypos_Ball = self.findobj(self.last_obs, courtXRng, courtYRng) # get x,y positions of ball
        xpos_Racket, ypos_Racket = self.findobj(self.last_obs, racketXRng, courtYRng) # get x,y positions of racket          
        #Now we know the position of racket relative to the ball. We can suggest the action for the racket so that it doesn't miss the ball.
        #For the time being, I am implementing a simple rule i.e. based on only the ypos of racket relative to the ball
        if ypos_Racket>ypos_Ball: #if the racket is lower than the ball the suggestion is to move up
          proposed_action = dconf['moves']['UP'] #move up
        elif ypos_Racket<ypos_Ball: #if the racket is higher than the ball the suggestion is to move down
          proposed_action = dconf['moves']['DOWN'] #move down
        elif ypos_Racket==ypos_Ball:
          proposed_action = dconf['moves']['NOMOVE'] #no move
        elif ypos_Ball==-1: #guess about proposed move can't be made because ball was not visible in the court
          proposed_action = -1 #no valid action guessed
        Images.append(np.sum(self.last_obs[courtYRng[0]:courtYRng[1],:,:],2))
        Ball_pos.append([courtXRng[0]-1+xpos_Ball,ypos_Ball])
        Racket_pos.append([racketXRng[0]-1+xpos_Racket,ypos_Racket])
      else:
        proposed_action = -1 #if there is no last_obs
        ypos_Ball = -1 #if there is no last_obs, no position of ball
        xpos_Ball = -1 #if there is no last_obs, no position of ball

      observation, reward, done, info = self.env.step(caction)

      #find position of ball after action
      xpos_Ball2, ypos_Ball2 = self.findobj(observation, courtXRng, courtYRng)        
      if xpos_Ball>0 and xpos_Ball2>0:
        if xpos_Ball2-xpos_Ball>0:
          ball_moves_towards_racket = 1 #use proposed action for reward only when the ball moves towards the racket
          current_ball_dir = 1 
        elif xpos_Ball2-xpos_Ball<0:
          ball_moves_towards_racket = 0
          current_ball_dir = -1
        else:
          ball_moves_towards_racket = 0
          current_ball_dir = 0 #direction can't be determinted  prob. because the ball didn't move in x dir.
      else:
        ball_moves_towards_racket = 0
        current_ball_dir = 0 #direction can't be determined because either current or last position of the ball is outside the court

      ball_hits_racket = 0
      if self.last_ball_dir==0 or current_ball_dir==0: # no way to find out if the ball hit the racket
        ball_hits_racket = 0 #therefore assumed that ball didn't hit the racket--weak/bad assumption
      else:
        if self.last_ball_dir==1 and current_ball_dir==-1 and reward==0:
          #if the ball was moving towards the racket and now its moving away from racket and didnt lose
          ball_hits_racket = 1
      self.last_ball_dir = current_ball_dir
      total_hits.append(ball_hits_racket) # i dont think this can be more than a single hit in 5 moves. so check if sum is greater than 1, print error

      self.env.render()
      self.last_obs = observation # current observation will be used as last_obs for the next action
      if done:
        self.env.reset()
        self.last_obs = [] # when the game ends, and new game starts, there is no last observation
        self.last_ball_dir=0
        done = False
      rewards.append(reward)
      proposed_actions.append(proposed_action)

      gray_Image = 255.0*rgb2gray(observation[courtYRng[0]:courtYRng[1],:,:]) # convert to grayscale; rgb2gray has 0-1 range so mul by 255
      gray_ds = downscale_local_mean(gray_Image,(8,8)) # then downsample
      gray_ds = np.where(gray_ds>np.min(gray_ds)+1,255,gray_ds) # Different thresholding
      lgimage.append(lgwght[adx]*gray_ds) # save weighted grayscale image from current frame
      self.countAll += 1

    # NB: previously we merged 2x2 pixels into 1 value. Now we merge 8x8 pixels into 1 value.
    # so the original 160x160 pixels will result into 20x20 values instead of previously used 80x80.        
    if len(lgimage)>1:
      dsum_Images = np.maximum(lgimage[0],lgimage[1])
      for gimage in lgimage[2:]: dsum_Images = np.maximum(dsum_Images,gimage)
    else:
      dsum_Images = lgimage[0]
    InputImages.append(dsum_Images) # save the input image

    dirs = self.computeMotion(dsum_Images) # compute directions of motion for every other pixel
    self.updateDirSensitiveRates(dirs) # update motion sensitive neuron input rates
    self.updateInputRates(dsum_Images) # update input rates to retinal neurons

    if done: # what is done? --- when done == 1, it means that 1 episode of the game ends, so it needs to be reset. 
      epCount.append(self.countAll)
      self.env.reset()
      self.env.frameskip = 3 
      self.countAll = 0 
    if np.sum(total_hits)>1:
      print('ERROR COMPUTING NUMBER OF HITS')
    for r in range(len(rewards)):
      if rewards[r]==-1: total_hits[r]=-1 #when the ball misses the racket, the reward is -1
    return rewards, epCount, InputImages, proposed_actions, total_hits, Racket_pos, Ball_pos, Images, dirs

  def playGameFake (self, epCount, InputImages): #actions are generated based on Vector Algebra
    actions = []
    rewards = []
    courtYRng, courtXRng, racketXRng = self.courtYRng, self.courtXRng, self.racketXRng # coordinate ranges for different objects (PONG-specific)      
    lgwght = np.linspace(0.6, 1, self.intaction) # time-decay grayscale image weights (earlier indices with lower weights are from older frames)
    lgimage = [] # grayscale images with decaying time-lagged input    
    for adx in range(self.intaction):
      if len(self.last_obs)==0:
        caction = random.randint(3,4)
      else:
        ImageCourt = self.last_obs[courtYRng[0]:courtYRng[1],courtXRng[0]:courtXRng[1],:]
        ImageAgent = self.last_obs[courtYRng[0]:courtYRng[1],racketXRng[0]:racketXRng[1],:]
        posBall = np.unravel_index(np.argmax(ImageCourt),ImageCourt.shape)
        posAgent = np.unravel_index(np.argmax(ImageAgent),ImageAgent.shape)
        yBall = posBall[0]
        yAgent = posAgent[0]
        if yBall>yAgent: caction = dconf['moves']['DOWN']
        elif yAgent>yBall: caction = dconf['moves']['UP']
        else: caction = dconf['moves']['NOMOVE']
      actions.append(caction)
      observation, reward, done, info = self.env.step(caction)
      self.last_obs = observation
      rewards.append(reward)
      gray_Image = 255.0*rgb2gray(observation[courtYRng[0]:courtYRng[1],:,:]) # convert to grayscale; rgb2gray has 0-1 range so mul by 255
      gray_ds = downscale_local_mean(gray_Image,(8,8)) # then downsample
      gray_ds = np.where(gray_ds>np.min(gray_ds)+1,255,gray_ds) # Different thresholding
      lgimage.append(lgwght[adx]*gray_ds) # save weighted grayscale image from current frame
      self.countAll += 1
    # NB: previously we merged 2x2 pixels into 1 value. Now we merge 8x8 pixels into 1 value.
    # so the original 160x160 pixels will result into 20x20 values instead of previously used 80x80.        
    if len(lgimage)>1:
      dsum_Images = np.maximum(lgimage[0],lgimage[1])
      for gimage in lgimage[2:]: dsum_Images = np.maximum(dsum_Images,gimage)
    else:
      dsum_Images = lgimage[0]
    InputImages.append(dsum_Images) # save the input image
    self.updateInputRates(dsum_Images) # update input rates to retinal neurons    
    self.env.render()
    if done:
      epCount.append(self.countAll)
      self.env.reset()
      self.env.frameskip = 3
      self.countAll = 0
      self.last_obs=[]
      self.last_bill_dir=0
    return rewards, actions, epCount, InputImages
            
