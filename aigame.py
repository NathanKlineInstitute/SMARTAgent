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

class AIGame:
    """ Interface to OpenAI gym game 
    """
    def __init__ (self,fcfg='sim.json'): # initialize variables
      self.env = env
      self.count = 0 
      self.countAll = 0
      self.ldir = ['E','NE','N', 'NW','W','SW','S','SE']
      self.ldirpop = ['EV1D'+Dir for Dir in self.ldir]
      self.lratepop = ['ER'] # populations that we calculate rates for
      for d in self.ldir: self.lratepop.append('EV1D'+d)
      self.dFVec = OrderedDict({pop:h.Vector() for pop in self.lratepop}) # NEURON Vectors for firing rate calculations
      self.dFiringRates = OrderedDict({pop:np.zeros(dconf['net'][pop]) for pop in self.lratepop}) # python objects for firing rate calculations
      self.dAngRange = OrderedDict({'EV1DE': (337,23),'EV1DNE': (22, 68), # angle ranges by population
                                    'EV1DN': (67, 113),'EV1DNW': (112, 158),
                                    'EV1DW': (157, 203),'EV1DSW': (202, 248),
                                    'EV1DS': (247, 293),'EV1DSE': (292, 338)})
      self.input_dim = int(np.sqrt(dconf['net']['ER']))
      self.dirSensitiveNeuronDim = 10 #int(0.5*self.input_dim)
      self.dirSensitiveNeuronRate = (0.0001, 30) # min, max firing rate (Hz) for dir sensitive neurons
      self.intaction = 5 # integrate this many actions together before returning reward information to model

    def updateInputRates (self, dsum_Images):
      # update input rates to retinal neurons
      InputImages.append(dsum_Images)
      #fr_Images = np.where(dsum_Images>1.0,100,dsum_Images) #Using this to check what number would work for firing rate
      #fr_Images = np.where(dsum_Images<10.0,0,dsum_Images)
      fr_Images = 40/(1+np.exp((np.multiply(-1,dsum_Images)+123)/25))
      fr_Images = np.subtract(fr_Images,7.722) #baseline firing rate subtraction. Instead all excitatory neurons are firing at 5Hz.
      #print(np.amax(fr_Images))
      self.dFiringRates['ER'] = np.reshape(fr_Images,400) #400 for 20*20
      
    def computeMotion (self, InputImages, dsum_Images):
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
          #print('Xinds',Rxs)
          #print('Yinds',Rys)
          FOV = np.zeros(shape=(len(Rxs),len(Rys)))
          for xinds in range(len(Rxs)):
            for yinds in range(len(Rys)):
              FOV[xinds,yinds] = dsum_Images[Rxs[xinds],Rys[yinds]]
          #print(FOV)
          max_value = np.amax(FOV)
          max_ind = np.where(FOV==max_value)
          #print('max inds', max_ind) 
          #since the most recent frame has highest pixel intensity, any pixel with the maximum intensity will be most probably the final instance of the object motion in that field of view
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
          #since latest frame has lowest pixel intensity (after ignoring background), any pixel with max intensity will most probably be first instance of object motion in that field of view
          if len(max_ind[0])>len(min_ind[0]):
            mL = len(min_ind[0])
          elif len(max_ind[0])<len(min_ind[0]):
            mL = len(max_ind[0])
          else:
            mL = len(max_ind[0])
          dir1 = [max_ind[0][range(mL)]-min_ind[0][range(mL)],max_ind[1][range(mL)]-min_ind[1][range(mL)]] #direction of the object motion in a field of view over last 5 frames/observations.
          dir2 = [np.median(dir1[1]),-1*np.median(dir1[0])] #flip y because indexing starts from top left.
          dirMain = [1,0] #using a reference for 0 degrees....considering first is for rows and second is for columns
          ndir2 = dir2 / np.linalg.norm(dir2)
          ndirMain = dirMain / np.linalg.norm(dirMain)
          theta = np.degrees(np.arccos(np.dot(ndir2,ndirMain))) #if theta is nan, no movement is detected
          if dir2[1]<0: theta = 360-theta 
          dirSensitiveNeurons[dSNeuronX,dSNeuronY] = theta
          if np.isnan(theta)=='False': print('Theta for FOV ',FOV,' is: ', theta)
      print('Computed angles:', dirSensitiveNeurons)
      dAngRange = self.dAngRange
      dInds = {pop:np.where(np.logical_and(dirSensitiveNeurons>dAngRange[pop][0],dirSensitiveNeurons<dAngRange[pop][1])) for pop in self.ldirpop}
      for pop in self.ldirpop:
        dtmp = self.dirSensitiveNeuronRate[0] * np.ones(shape=(dirSensitiveNeuronDim,dirSensitiveNeuronDim))
        dtmp[dInds[pop]] = self.dirSensitiveNeuronRate[1] # firing rate for active dir sensitive neurons; later could include noise.
        self.dFiringRates[pop] = np.reshape(dtmp , 100) # this assumes 100 neurons in that population
      
    ################################
    ### PLAY GAME
    ###############################
    def playGame (self, actions, epCount, InputImages, last_obs, last_ball_dir): #actions need to be generated from motor cortex
      rewards = []
      proposed_actions =[]
      total_hits = []
      input_dim = self.input_dim
      # previously we merged 2x2 pixels into 1 value. Now we merge 8x8 pixels into 1 value.
      # so the original 160x160 pixels will result into 20x20 values instead of previously used 80x80.
      dsum_Images = np.zeros(shape=(input_dim,input_dim)) 
      gray_Image = np.zeros(shape=(160,160))
      done = False

      for a in range(self.intaction):
        #for each action generated by the firing rate of the motor cortex, find the suggested-action by comparing the position of the ball and racket 
        caction = actions[a] #action generated by the firing rate of the motor cortex

        if np.shape(last_obs)[0]>0: #if last_obs is not empty              
          ImageCourt = last_obs[34:194,20:140,:] # what is significance of indices 34 through 194?
          ImageAgent = last_obs[34:194,141:144,:]
          sIC = np.sum(ImageCourt,2) #since the color of object is uniform, add values or r,g,b to get a single value
          sIA = np.sum(ImageAgent,2) #since the color of object is uniform, add values or r,g,b to get a single value
          pixelVal_Agent = np.amax(sIA) #find the pixel value representing Agent/Racket
          pixelVal_Ball = np.amax(sIC) #find the pixel value representing Ball..court is blackish
          sIA[sIA<pixelVal_Agent]=0 #make binary image of Agent/Racket
          sIC[sIC<pixelVal_Ball]=0 #make binary image of court
          Ball_inds = []
          for i in range(sIC.shape[0]):
            for j in range(sIC.shape[1]):
              if sIC[i,j]>0:
                Ball_inds.append([i,j])
          if sIC.shape[0]*sIC.shape[1]==np.shape(Ball_inds)[0]: #if there is no ball in the court
            ypos_Ball = -1
            xpos_Ball = -1
          else:
            ypos_Ball = np.median(Ball_inds,0)[0] #y position of the center of mass of the ball
            xpos_Ball = np.median(Ball_inds,0)[1] #x position of the center of mass of the ball
          Racket_inds = []
          for i in range(sIA.shape[0]):
            for j in range(sIA.shape[1]):
              if sIA[i,j]>0:
                Racket_inds.append([i,j])
          ypos_Racket = np.median(Racket_inds,0)[0] #y position of the center of mass of the racket
          xpos_Racket = np.median(Racket_inds,0)[1] #x position of the center of mass of the racket
          #Now we know the position of racket relative to the ball. We can suggest the action for the racket so that it doesn't miss the ball.
          #For the time being, I am implementing a simple rule i.e. based on only the ypos of racket relative to the ball
          if ypos_Racket>ypos_Ball: #if the racket is lower than the ball the suggestion is to move up
            proposed_action = 4 #move up
          elif ypos_Racket<ypos_Ball: #if the racket is higher than the ball the suggestion is to move down
            proposed_action = 3 #move down
          elif ypos_Racket==ypos_Ball:
            proposed_action = 1 #no move
          elif ypos_Ball==-1: #guess about proposed move can't be made because ball was not visible in the court
            proposed_action = -1 #no valid action guessed
        else:
          proposed_action = -1 #if there is no last_obs
          ypos_Ball = -1 #if there is no last_obs, no position of ball
          xpos_Ball = -1 #if there is no last_obs, no position of ball

        observation, reward, done, info = self.env.step(caction)

        #find position of ball after action
        ImageCourt2 = observation[34:194,20:140,:]
        sIC2 = np.sum(ImageCourt2,2) #since the color of object is uniform, add values or r,g,b to get a single value
        newpixelVal_Ball = np.amax(sIC2) #find the pixel value representing Ball..court is blackish
        sIC2[sIC2<newpixelVal_Ball]=0 #make binary image of court
        Ball2_inds = []
        for i in range(sIC2.shape[0]):
          for j in range(sIC2.shape[1]):
            if sIC2[i,j]>0:
              Ball2_inds.append([i,j])
        if sIC2.shape[0]*sIC2.shape[1]==np.shape(Ball2_inds)[0]: #if there is no ball in the court
          ypos_Ball2 = -1
          xpos_Ball2 = -1
        else:
          ypos_Ball2 = np.median(Ball2_inds,0)[0] #y position of the center of mass of the ball
          xpos_Ball2 = np.median(Ball2_inds,0)[1] #x position of the center of mass of the ball
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
        if last_ball_dir==0 or current_ball_dir==0: # no way to find out if the ball hit the racket
          ball_hits_racket = 0 #therefore assumed that ball didn't hit the racket--weak/bad assumption
        else:
          if last_ball_dir==1 and current_ball_dir==-1 and reward==0:
            #if the ball was moving towards the racket and now its moving away from racket and didnt lose
            ball_hits_racket = 1
        last_ball_dir = current_ball_dir
        total_hits.append(ball_hits_racket) # i dont think this can be more than a single hit in 5 moves. so check if sum is greater than 1, print error

        self.env.render()
        last_obs = observation #current observation will be used as last_obs for the next action
        if done:
          self.env.reset()
          last_obs = [] # when the game ends, and new game starts, there is no last observation
          done = False
        rewards.append(reward)
        proposed_actions.append(proposed_action)

        Image = observation[34:194,:,:] # why does it only use rows 34 through 194?
        for i in range(160):
          for j in range(160):
            gray_Image[i][j]= 0.2989*Image[i][j][0] + 0.5870*Image[i][j][1] + 0.1140*Image[i][j][2]
        gray_ds = downscale_local_mean(gray_Image,(8,8))
        gray_ds = np.where(gray_ds>np.min(gray_ds)+1,255,gray_ds) #Different thresholding
        if self.count==0: # 
          i0 = 0.6*gray_ds
          self.count = self.count+1
        elif self.count==1:
          i1 = 0.7*gray_ds
          self.count = self.count+1
        elif self.count==2:
          i2 = 0.8*gray_ds
          self.count = self.count+1
        elif self.count==3:
          i3 = 0.9*gray_ds
          self.count = self.count+1
        else:
          i4 = 1.0*gray_ds
          self.count = 0
        self.countAll = self.countAll+1

      dsum_Images = np.maximum(i0,i1)
      dsum_Images = np.maximum(dsum_Images,i2)
      dsum_Images = np.maximum(dsum_Images,i3)
      dsum_Images = np.maximum(dsum_Images,i4)

      self.computeMotion(dsum_Images) # compute directions of motion for every other pixel and update motion sensitive neuron input rates
      self.updateInputRates(InputImages,dsum_Images) # update input rates to retinal neurons

      if done: # what is done? --- when done == 1, it means that 1 episode of the game ends, so it needs to be reset. 
        epCount.append(self.countAll)
        self.env.reset()
        self.env.frameskip = 3 
        self.countAll = 0 # should self.count also get set to 0?
      if np.sum(total_hits)>1:
        print('ERROR COMPUTING NUMBER OF HITS')
      for r in range(len(rewards)):
        if rewards[r]==-1: total_hits[r]=-1 #when the ball misses the racket, the reward is -1
      return rewards, epCount, InputImages, last_obs, proposed_actions, last_ball_dir, total_hits

  def playGameFake (self, last_obs, epCount, InputImages): #actions are generated based on Vector Algebra
      #rewards = np.zeros(shape=(1,5))
      actions = []
      rewards = []
      dsum_Images = np.zeros(shape=(20,20)) #previously we merged 2x2 pixels into 1 value. Now we merge 8x8 pixels into 1 value. so the original 160x160 pixels will result into 20x20 values instead of previously used 80x80.
      #print(actions)
      for a in range(5):
          #action = random.randint(3,4)
          if len(last_obs)==0:
              caction = random.randint(3,4)
          else:
              ImageCourt = last_obs[34:194,20:140,:]
              ImageAgent = last_obs[34:194,141:144,:]
              posBall = np.unravel_index(np.argmax(ImageCourt),ImageCourt.shape)
              posAgent = np.unravel_index(np.argmax(ImageAgent),ImageAgent.shape)
              yBall = posBall[0]
              yAgent = posAgent[0]
              if yBall>yAgent: #Move down
                  caction = 3    
              elif yAgent>yBall: #Move up
                  caction = 4
              else: #Dont move
                  caction = 1
          actions.append(caction)
          observation, reward, done, info = self.env.step(caction)
          last_obs = observation
          rewards.append(reward)
          Image = observation[34:194,:,:]
          gray_Image = np.zeros(shape=(160,160))
          for i in range(160):
              for j in range(160):
                  gray_Image[i][j]= 0.2989*Image[i][j][0] + 0.5870*Image[i][j][1] + 0.1140*Image[i][j][2]
          #gray_ds = gray_Image[:,range(0,gray_Image.shape[1],2)]
          #gray_ds = gray_ds[range(0,gray_ds.shape[0],2),:]
          gray_ds = downscale_local_mean(gray_Image,(8,8))
          gray_ds = np.where(gray_ds>np.min(gray_ds)+1,255,gray_ds) #Different thresholding
          #gray_ds = np.where(gray_ds<127,0,gray_ds)
          #gray_ds = np.where(gray_ds>=127,255,gray_ds)
          if self.count==0:
              i0 = 0.6*gray_ds
              self.count = self.count+1
          elif self.count==1:
              i1 = 0.7*gray_ds
              self.count = self.count+1
          elif self.count==2:
              i2 = 0.8*gray_ds
              self.count = self.count+1
          elif self.count==3:
              i3 = 0.9*gray_ds
              self.count = self.count+1
          else:
              i4 = 1.0*gray_ds
              self.count = 0
          self.countAll = self.countAll+1
      dsum_Images = np.maximum(i0,i1)
      dsum_Images = np.maximum(dsum_Images,i2)
      dsum_Images = np.maximum(dsum_Images,i3)
      dsum_Images = np.maximum(dsum_Images,i4)
      InputImages.append(dsum_Images)
      #fr_Images = np.where(dsum_Images>1.0,100,dsum_Images) #Using this to check what number would work for firing rate
      #fr_Images = np.where(dsum_Images<10.0,0,dsum_Images)
      fr_Images = 40/(1+np.exp((np.multiply(-1,dsum_Images)+123)/25))
      fr_Images = np.subtract(fr_Images,7.722) #baseline firing rate subtraction. Instead all excitatory neurons are firing at 5Hz.
      #print(np.amax(fr_Images))
      self.firing_rates = np.reshape(fr_Images,400) #6400 for 80*80 Image, now its 400 for 20*20
      self.env.render()
      #print(self.countAll)
      if done:
          epCount.append(self.countAll)
          self.env.reset()
          self.env.frameskip = 3
          self.countAll = 0
      return rewards, actions, last_obs, epCount, InputImages
            
