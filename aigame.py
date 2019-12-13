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
#comment 

import gym
env = gym.make("Pong-v0")
env.reset()
  
class SMARTAgent:
    def __init__(self): # initialize variables
        self.env = env
        self.count = 0 
        self.countAll = 0
    ################################
    ### PLAY GAME
    ###############################
    def playGame(self, actions): #actions need to be generated from motor cortex
        #rewards = numpy.zeros(shape=(1,5))
        rewards = []
        dsum_Images = numpy.zeros(shape=(80,80))
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
            gray_ds = gray_Image[:,range(0,gray_Image.shape[1],2)]
            gray_ds = gray_ds[range(0,gray_ds.shape[0],2),:]
            gray_ds = numpy.where(gray_ds<127,0,gray_ds)
            gray_ds = numpy.where(gray_ds>=127,255,gray_ds)
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
        #fr_Images = numpy.where(dsum_Images>1.0,100,dsum_Images) #Using this to check what number would work for firing rate
        #fr_Images = numpy.where(dsum_Images<10.0,0,dsum_Images)
        fr_Images = 40/(1+numpy.exp((numpy.multiply(-1,dsum_Images)+123)/25))
        #print(numpy.amax(fr_Images))
        self.firing_rates = numpy.reshape(fr_Images,6400)
        self.env.render()
        print(self.countAll)
        if self.countAll==1000:
            self.env.reset()
            self.countAll = 0
        return rewards
        #return firing_rates

    ################################          
    ### RUN     
    ################################
    def run(self, t, f):
        #SMARTAgent.playGame(self)
        cind = 0
        for cell in [c for c in f.net.cells]:   # shoulder
            for stim in cell.stims:
                if stim['source'] == 'stimMod':
                    if self.firing_rates[cind]>1000.0:
                        print(self.firing_rates[cind])
                    stim['hObj'].interval = 1000.0/self.firing_rates[cind] # interval in ms as a function of rate
                    if self.firing_rates[cind]>40.0:
                        print(self.firing_rates[cind])
            cind = cind+1
                      





