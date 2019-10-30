"""
MSARM
Code to connect a virtual arm (simple kinematic arm implemented in python) to the M1 model
Adapted from arm.hoc in arm2dms
Version: 2015jan28 by salvadordura@gmail.com
"""

from neuron import h
from numpy import exp
from pylab import concatenate, figure, show, ion, ioff, pause,xlabel, ylabel, plot, Circle, sqrt, arctan, arctan2, close
from copy import copy
from random import uniform, seed, sample, randint


  
class Arm:

    ################################          
    ### RUN     
    ################################
    def run(t, f):
        # Update cells response based on virtual arm proprioceptive feedback (angles)
        for cell in [c for c in f.net.cells]:   # shoulder
            for stim in cell.stims:
                if stim['source'] == 'stimMod':
                    stim['hObj'].interval = 100.0/(1/(1+exp((-t+10)/0.25))) # interval in ms as a function of rate
                      





