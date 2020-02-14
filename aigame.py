"""
SMARTAgent
Code to connect a open ai gym game to the SMARTAgent model (V1-M1-RL)
Adapted from arm.py
Original Version: 2015jan28 by salvadordura@gmail.com
Modified Version: 2019oct1 by haroon.anwar@gmail.com
"""

from neuron import h
from numpy import exp
from pylab import concatenate, figure, show, ion, ioff, pause,xlabel, ylabel, plot, Circle, sqrt, arctan, arctan2, close
from copy import copy
from random import uniform, seed, sample, randint

from matplotlib import pyplot as plt
import random
import numpy

from skimage.transform import downscale_local_mean

#comment 


import gym
env = gym.make("Pong-v0")
env.reset()
  
class SMARTAgent:
    def __init__(self): # initialize variables
        self.env = env
        self.count = 0 
        self.countAll = 0
        self.fvec = h.Vector()
        self.firing_rates = numpy.zeros(shape=(1,400))
    ################################
    ### PLAY GAME
    ###############################
    def playGame(self, actions, epCount, InputImages): #actions need to be generated from motor cortex
        #rewards = numpy.zeros(shape=(1,5))
        rewards = []
        dsum_Images = numpy.zeros(shape=(20,20)) #previously we merged 2x2 pixels into 1 value. Now we merge 8x8 pixels into 1 value. so the original 160x160 pixels will result into 20x20 values instead of previously used 80x80.
        #print(actions)
        for a in range(5):
            #action = random.randint(3,4)
            caction = actions[a]
            observation, reward, done, info = self.env.step(caction)
            rewards.append(reward)
            Image = observation[34:194,:,:]
            gray_Image = numpy.zeros(shape=(160,160))
            for i in range(160):
                for j in range(160):
                    gray_Image[i][j]= 0.2989*Image[i][j][0] + 0.5870*Image[i][j][1] + 0.1140*Image[i][j][2]
            #gray_ds = gray_Image[:,range(0,gray_Image.shape[1],2)]
            #gray_ds = gray_ds[range(0,gray_ds.shape[0],2),:]
            gray_ds = downscale_local_mean(gray_Image,(8,8))
            gray_ds = numpy.where(gray_ds>numpy.min(gray_ds)+1,255,gray_ds) #Different thresholding
            #gray_ds = numpy.where(gray_ds<127,0,gray_ds)
            #gray_ds = numpy.where(gray_ds>=127,255,gray_ds)
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
        dsum_Images = numpy.maximum(i0,i1)
        dsum_Images = numpy.maximum(dsum_Images,i2)
        dsum_Images = numpy.maximum(dsum_Images,i3)
        dsum_Images = numpy.maximum(dsum_Images,i4)
        InputImages.append(dsum_Images)
        #fr_Images = numpy.where(dsum_Images>1.0,100,dsum_Images) #Using this to check what number would work for firing rate
        #fr_Images = numpy.where(dsum_Images<10.0,0,dsum_Images)
        fr_Images = 40/(1+numpy.exp((numpy.multiply(-1,dsum_Images)+123)/25))
        fr_Images = numpy.subtract(fr_Images,7.722) #baseline firing rate subtraction. Instead all excitatory neurons are firing at 5Hz.
        #print(numpy.amax(fr_Images))
        self.firing_rates = numpy.reshape(fr_Images,400) #6400 for 80*80 Image, now its 400 for 20*20
        self.env.render()
        print(self.countAll)
        if done:
            epCount.append(self.countAll)
            self.env.reset()
            self.countAll = 0
        return rewards, epCount, InputImages
        #return firing_rates

    def playGameFake(self, last_obs, epCount, InputImages): #actions are generated based on Vector Algebra
        #rewards = numpy.zeros(shape=(1,5))
        actions = []
        rewards = []
        dsum_Images = numpy.zeros(shape=(20,20)) #previously we merged 2x2 pixels into 1 value. Now we merge 8x8 pixels into 1 value. so the original 160x160 pixels will result into 20x20 values instead of previously used 80x80.
        #print(actions)
        for a in range(5):
            #action = random.randint(3,4)
            if len(last_obs)==0:
                caction = random.randint(3,4)
            else:
                ImageCourt = last_obs[34:194,20:140,:]
                ImageAgent = last_obs[34:194,141:144,:]
                posBall = numpy.unravel_index(numpy.argmax(ImageCourt),ImageCourt.shape)
                posAgent = numpy.unravel_index(numpy.argmax(ImageAgent),ImageAgent.shape)
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
            gray_Image = numpy.zeros(shape=(160,160))
            for i in range(160):
                for j in range(160):
                    gray_Image[i][j]= 0.2989*Image[i][j][0] + 0.5870*Image[i][j][1] + 0.1140*Image[i][j][2]
            #gray_ds = gray_Image[:,range(0,gray_Image.shape[1],2)]
            #gray_ds = gray_ds[range(0,gray_ds.shape[0],2),:]
            gray_ds = downscale_local_mean(gray_Image,(8,8))
            gray_ds = numpy.where(gray_ds>numpy.min(gray_ds)+1,255,gray_ds) #Different thresholding
            #gray_ds = numpy.where(gray_ds<127,0,gray_ds)
            #gray_ds = numpy.where(gray_ds>=127,255,gray_ds)
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
        dsum_Images = numpy.maximum(i0,i1)
        dsum_Images = numpy.maximum(dsum_Images,i2)
        dsum_Images = numpy.maximum(dsum_Images,i3)
        dsum_Images = numpy.maximum(dsum_Images,i4)
        InputImages.append(dsum_Images)
        #fr_Images = numpy.where(dsum_Images>1.0,100,dsum_Images) #Using this to check what number would work for firing rate
        #fr_Images = numpy.where(dsum_Images<10.0,0,dsum_Images)
        fr_Images = 40/(1+numpy.exp((numpy.multiply(-1,dsum_Images)+123)/25))
        fr_Images = numpy.subtract(fr_Images,7.722) #baseline firing rate subtraction. Instead all excitatory neurons are firing at 5Hz.
        #print(numpy.amax(fr_Images))
        self.firing_rates = numpy.reshape(fr_Images,400) #6400 for 80*80 Image, now its 400 for 20*20
        self.env.render()
        print(self.countAll)
        if done:
            epCount.append(self.countAll)
            self.env.reset()
            self.countAll = 0
        return rewards, actions, last_obs, epCount, InputImages

    ################################          
    ### RUN     
    ################################
    def run(self, t, f):
        #SMARTAgent.playGame(self)
        if f.rank==0:
            f.pc.broadcast(self.fvec.from_python(self.firing_rates),0)
        else:
            f.pc.broadcast(self.fvec,0)
            firing_rates = self.fvec.to_python()
        cind = 0
        for cell in [c for c in f.net.cells]:   
            for stim in cell.stims:
                if stim['source'] == 'stimMod':
                    stim['hObj'].interval = 1000.0/self.firing_rates[cind] # interval in ms as a function of rate
            cind = cind+1
                      





