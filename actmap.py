import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from conf import dconf
from collections import OrderedDict
from pylab import *
import os
import anim
from matplotlib import animation
from simdat import loadInputImages, loadsimdat, loadMotionFields, loadObjPos
from imgutils import getoptflow, getoptflowframes

rcParams['font.size'] = 12

simConfig, pdf, actreward, dstartidx, dendidx, dnumc, dspkID, dspkT, InputImages, ldflow = loadsimdat(dconf['sim']['name'])



#fig, axs, plt = animActivityMaps('test2.mp4')
# fig, axs, plt = animActivityMaps('gif/'+dconf['sim']['name']+'actmap.mp4', framerate=10)

