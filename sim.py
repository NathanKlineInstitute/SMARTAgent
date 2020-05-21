from netpyne import specs, sim
from neuron import h
import numpy as np
import random
from conf import dconf # configuration dictionary
import pandas as pd
import pickle
from collections import OrderedDict
from connUtils import *
from matplotlib import pyplot as plt
import os
import time
import anim
from matplotlib import animation

random.seed(1234) # this will not work properly across runs with different number of nodes

sim.davgW = {} # average adjustable weights on a target population
sim.allTimes = []
sim.allRewards = [] # list to store all rewards
sim.allActions = [] # list to store all actions
sim.allProposedActions = [] # list to store all proposed actions
sim.allHits = [] #list to store all hits
sim.allMotorOutputs = [] # list to store firing rate of output motor neurons.
sim.ActionsRewardsfilename = 'data/'+dconf['sim']['name']+'ActionsRewards.txt'
sim.MotorOutputsfilename = 'data/'+dconf['sim']['name']+'MotorOutputs.txt'
sim.WeightsRecordingTimes = []
sim.allRLWeights = [] # list to store weights --- should remove that
sim.allNonRLWeights = [] # list to store weights --- should remove that
#sim.NonRLweightsfilename = 'data/'+dconf['sim']['name']+'NonRLweights.txt'  # file to store weights
sim.plotWeights = 0  # plot weights
sim.saveWeights = 1  # save weights
sim.saveInputImages = 1 #save Input Images (5 game frames)
sim.saveMotionFields = 1 # whether to save the motion fields
recordWeightStepSize = dconf['sim']['recordWeightStepSize']
normalizeWeightStepSize = dconf['sim']['normalizeWeightStepSize']
#recordWeightDT = 1000 # interval for recording synaptic weights (change later)
recordWeightDCells = 1 # to record weights for sub samples of neurons
tstepPerAction = dconf['sim']['tstepPerAction'] # time step per action (in ms)

fid4=None # only used by rank 0

scale = dconf['net']['scale']
ETypes = ['ER','EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE','EV4','EMT', 'EMDOWN', 'EMUP']
#ITypes = ['IR','IV1','IV1D','IV4','IMT','IM'] #
ITypes = ['IR','IV1','IV4','IMT','IM'] # 
#allpops = ['ER','IR','EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE','IV1','IV1D','EV4','IV4','EMT','IMT','EMDOWN','EMUP','IM']
allpops = ['ER','IR','EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE','IV1','EV4','IV4','EMT','IMT','EMDOWN','EMUP','IM']
#EDirPops = ['EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE']
#IDirPops = ['IV1D']
EMotorPops = ['EMDOWN', 'EMUP'] # excitatory neuron motor populations
dnumc = OrderedDict({ty:dconf['net'][ty]*scale for ty in allpops}) # number of neurons of a given type

# Network parameters
netParams = specs.NetParams() #object of class NetParams to store the network parameters
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells

#Population parameters
for ty in allpops:
  if ty in ETypes:
    netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': 'Mainen'}
  else:
    netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': 'FS_BasketCell'}
    
netParams.importCellParams(label='PYR_Mainen_rule', conds={'cellType': ETypes}, fileName='cells/mainen.py', cellName='PYR2')
netParams.importCellParams(label='FS_BasketCell_rule', conds={'cellType': ITypes}, fileName='cells/FS_BasketCell.py', cellName='Bas')

netParams.cellParams['PYR_Mainen_rule']['secs']['soma']['threshold'] = 0.0
netParams.cellParams['FS_BasketCell_rule']['secs']['soma']['threshold'] = -10.0

## Synaptic mechanism parameters
netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}  # excitatory synaptic mechanism
netParams.synMechParams['NMDA'] = {'mod': 'Exp2Syn', 'tau1': 0.15, 'tau2': 166.0, 'e': 0} # NMDA
netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80}  # inhibitory synaptic mechanism

#wmin should be set to the initial/baseline weight of the connection.
STDPparams = {'hebbwt': 0.0001, 'antiwt':-0.00001, 'wbase': 0.0012, 'wmax': 50, 'RLon': 0 , 'RLhebbwt': 0.001, 'RLantiwt': -0.000,
        'tauhebb': 10, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 0, 'verbose':0}


dSTDPparamsRL = {} # STDP-RL parameters for AMPA,NMDA synapses; generally uses shorter/longer eligibility traces
for sy in ['AMPA', 'NMDA']: dSTDPparamsRL[sy] = dconf['RL'][sy]

# these are the image-based inputs provided to the R (retinal) cells
netParams.stimSourceParams['stimMod'] = {'type': 'NetStim', 'rate': 'variable', 'noise': 0}
netParams.stimTargetParams['stimMod->R'] = {'source': 'stimMod',
        'conds': {'pop': 'ER'},
        'convergence': 1,
        'weight': 0.05,
        'delay': 1,
        'synMech': 'AMPA'}
netParams.stimTargetParams['stimMod->DirSelInput'] = {'source': 'stimMod',
        'conds': {'pop': ['EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE']},
        'convergence': 1,
        'weight': 0.01,
        'delay': 1,
        'synMech': 'AMPA'}
#background input to inhibitory neurons to increase their firing rate

# Stimulation parameters

""" 
# weights are currently at 0 so do not need to simulate them
netParams.stimSourceParams['ebkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['ebkg->all'] = {'source': 'ebkg', 'conds': {'cellType': ['EV1','EV4','EMT']}, 'weight': 0.0, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'}

netParams.stimSourceParams['MLbkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['MLbkg->all'] = {'source': 'MLbkg', 'conds': {'cellType': ['EMDOWN']}, 'weight': 0.0, 'delay': 1, 'synMech': 'AMPA'}

netParams.stimSourceParams['MRbkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['MRbkg->all'] = {'source': 'MRbkg', 'conds': {'cellType': ['EMUP']}, 'weight': 0.0, 'delay': 1, 'synMech': 'AMPA'}

netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 20, 'noise': 1.0}
netParams.stimTargetParams['bkg->all'] = {'source': 'bkg', 'conds': {'cellType': ['IR','IV1','IV4','IMT']}, 'weight': 0.0, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'}
"""

######################################################################################

#####################################################################################
#Feedforward excitation
#E to E - Feedforward connections
blistERtoEV1 = connectLayerswithOverlap(NBpreN = dnumc['ER'], NBpostN = dnumc['EV1'], overlap_xdir = 3)
blistEV1toEV4 = connectLayerswithOverlap(NBpreN = dnumc['EV1'], NBpostN = dnumc['EV4'], overlap_xdir = 3)
blistEV4toEMT = connectLayerswithOverlap(NBpreN = dnumc['EV4'], NBpostN = dnumc['EMT'], overlap_xdir = 3) #was 15
#blistITtoMI = connectLayerswithOverlap(NBpreN = NB_ITneurons, NBpostN = NB_MIneurons, overlap_xdir = 3) #Not sure if this is a good strategy instead of all to all
#blistMItoMO = connectLayerswithOverlap(NBpreN = NB_MIneurons, NBpostN = NB_MOneurons, overlap_xdir = 3) #was 19
#blistMItoMO: Feedforward for MI to MO is all to all and can be specified in the connection statement iteself

#E to I - Feedforward connections
#blistERtoIV1 = connectLayerswithOverlap(NBpreN = dnumc['ER'], NBpostN = dnumc['IV1'], overlap_xdir = 3)
#blistEV1toIV4 = connectLayerswithOverlap(NBpreN = dnumc['EV1'], NBpostN = dnumc['IV4'], overlap_xdir = 3) 
#blistEV4toIMT = connectLayerswithOverlap(NBpreN = dnumc['EV4'], NBpostN = dnumc['IMT'], overlap_xdir = 3) 

#E to I - WithinLayer connections
blistERtoIR = connectLayerswithOverlap(NBpreN = dnumc['ER'], NBpostN = dnumc['IR'], overlap_xdir = 3)
blistEV1toIV1 = connectLayerswithOverlap(NBpreN = dnumc['EV1'], NBpostN = dnumc['IV1'], overlap_xdir = 3)
#blistEV1DtoIV1D = connectLayerswithOverlap(NBpreN = dnumc['EV1DE'], NBpostN = dnumc['IV1D'], overlap_xdir = 3) # for dir selective E -> I
blistEV4toIV4 = connectLayerswithOverlap(NBpreN = dnumc['EV4'], NBpostN = dnumc['IV4'], overlap_xdir = 3)
blistEMTtoIMT = connectLayerswithOverlap(NBpreN = dnumc['EMT'], NBpostN = dnumc['IMT'], overlap_xdir = 3)

#I to E - WithinLayer Inhibition
blistIRtoER = connectLayerswithOverlapDiv(NBpreN = dnumc['IR'], NBpostN = dnumc['ER'], overlap_xdir = 5)
blistIV1toEV1 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV1'], NBpostN = dnumc['EV1'], overlap_xdir = 5)
#blistIV1DtoEV1D = connectLayerswithOverlapDiv(NBpreN = dnumc['IV1D'], NBpostN = dnumc['EV1DE'], overlap_xdir = 5) # for dir selective I -> E
blistIV4toEV4 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV4'], NBpostN = dnumc['EV4'], overlap_xdir = 5)
blistIMTtoEMT = connectLayerswithOverlapDiv(NBpreN = dnumc['IMT'], NBpostN = dnumc['EMT'], overlap_xdir = 5)

#Feedbackward excitation
#E to E  
blistEV1toER = connectLayerswithOverlapDiv(NBpreN = dnumc['EV1'], NBpostN = dnumc['ER'], overlap_xdir = 3)
blistEV4toEV1 = connectLayerswithOverlapDiv(NBpreN = dnumc['EV4'], NBpostN = dnumc['EV1'], overlap_xdir = 3)
blistEMTtoEV4 = connectLayerswithOverlapDiv(NBpreN = dnumc['EMT'], NBpostN = dnumc['EV4'], overlap_xdir = 3)

#Feedforward inhibition
#I to I
blistIV1toIV4 = connectLayerswithOverlap(NBpreN = dnumc['IV1'], NBpostN = dnumc['IV4'], overlap_xdir = 5)
blistIV4toIMT = connectLayerswithOverlap(NBpreN = dnumc['IV4'], NBpostN = dnumc['IMT'], overlap_xdir = 5)

#Feedbackward inhibition
#I to E 
blistIV1toER = connectLayerswithOverlapDiv(NBpreN = dnumc['IV1'], NBpostN = dnumc['ER'], overlap_xdir = 5)
blistIV4toEV1 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV4'], NBpostN = dnumc['EV1'], overlap_xdir = 5)
blistIMTtoEV4 = connectLayerswithOverlapDiv(NBpreN = dnumc['IMT'], NBpostN = dnumc['EV4'], overlap_xdir = 5)

#Simulation options
simConfig = specs.SimConfig()           # object of class SimConfig to store simulation configuration

simConfig.duration = dconf['sim']['duration'] # 100e3 # 0.1e5                      # Duration of the simulation, in ms
simConfig.dt = dconf['sim']['dt']                            # Internal integration timestep to use
simConfig.hParams['celsius'] = 37 # make sure temperature is set. otherwise we're at squid temperature
simConfig.verbose = dconf['sim']['verbose']                       # Show detailed messages
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordCellsSpikes = [-1]
simConfig.recordStep = dconf['sim']['recordStep'] # Step size in ms to save data (e.g. V traces, LFP, etc)
simConfig.filename = 'data/'+dconf['sim']['name']+'simConfig'  # Set file output name
simConfig.saveJson = False
simConfig.savePickle = True            # Save params, network and sim output to pickle file
simConfig.saveMat = False
simConfig.saveFolder = 'data'
# simConfig.backupCfg = ['sim.json', 'backupcfg/'+dconf['sim']['name']+'sim.json']

#simConfig.analysis['plotRaster'] = True                         # Plot a raster
# ['ER','IR','EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE','IV1','EV4','IV4','EMT','IMT','EMDOWN','EMUP','IM']
#simConfig.analysis['plotTraces'] = {'include': [(pop, 0) for pop in allpops]}
#simConfig.analysis['plotTraces'] = {'include': [(pop, 0) for pop in ['ER','IR','EV1','EV1DE','IV1','EV4','IV4','EMT','IMT','EMDOWN','IM']]}
simConfig.analysis['plotTraces'] = {'include': [(pop, 0) for pop in ['ER','IR','EV1','EV1DE','IV1','EMDOWN','EMUP','IM']]}

#simConfig.analysis['plotRaster'] = {'timeRange': [500,1000],'popRates':'overlay','saveData':'data/RasterData.pkl','showFig':True}
#simConfig.analysis['plotRaster'] = {'popRates':'overlay','saveData':'data/'+dconf['sim']['name']+'RasterData.pkl','showFig':dconf['sim']['doplot']}
simConfig.analysis['plotRaster'] = {'popRates':'overlay','showFig':dconf['sim']['doplot']}
#simConfig.analysis['plot2Dnet'] = True 
#simConfig.analysis['plotConn'] = True           # plot connectivity matrix

# synaptic weight gain (based on E, I types)
cfg = simConfig
cfg.EEGain = 0.75  # E to E scaling factor
cfg.EIGain = 1.0 # E to I scaling factor
cfg.IEGain = 10.0 # I to E scaling factor
cfg.IIGain = 10.0  # I to I scaling factor

### from https://www.neuron.yale.edu/phpBB/viewtopic.php?f=45&t=3770&p=16227&hilit=memory#p16122
cfg.saveCellSecs = bool(dconf['sim']['saveCellSecs']) # if False removes all data on cell sections prior to gathering from nodes
cfg.saveCellConns = bool(dconf['sim']['saveCellConns']) # if False removes all data on cell connections prior to gathering from nodes
#cfg.gatherOnlySimData = True # do not set to True, when True gathers from nodes only the output simulation data (not the network instance)
###

"""
recWeight = 0.0001 #weight for recurrent connections within each area.
recProb = 0.2 #probability of recurrent connections within each area.
#Local excitation
#E to E - may want plasticity between EMDOWN<>EMDOWN and EMUP<>EMUP
for epop in ['ER', 'EV1', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS','EV1DSE','EV4','EMT','EMDOWN','EMUP']:
  netParams.connParams[epop+'->'+epop] = {
    'preConds': {'pop': epop},
    'postConds': {'pop': epop},
    'probability': recProb,
    'weight': recWeight * cfg.EEGain, #0.0001
    'delay': 2,
    'synMech': 'AMPA', 'sec':'dend', 'loc':0.5}
"""  
             
#E to I within area
netParams.connParams['ER->IR'] = {
        'preConds': {'pop': 'ER'},
        'postConds': {'pop': 'IR'},
        'connList': blistERtoIR,
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
netParams.connParams['EV1->IV1'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'IV1'},
        'connList': blistEV1toIV1,
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}

"""
for prety in EDirPops:
  for poty in IDirPops:
    netParams.connParams[prety+'->'+poty] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'connList': blistEV1DtoIV1D,
      'weight': 0.02 * cfg.EIGain,
      'delay': 2,
      'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
"""

netParams.connParams['EV4->IV4'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'IV4'},
        'connList': blistEV4toIV4,
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
netParams.connParams['EMT->IMT'] = {
        'preConds': {'pop': 'EMT'},
        'postConds': {'pop': 'IMT'},
        'connList': blistEMTtoIMT,
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
netParams.connParams['EMDOWN->IM'] = {
        'preConds': {'pop': 'EMDOWN'},
        'postConds': {'pop': 'IM'},
        #'probability': 0.125/2.,
        'convergence': prob2conv(0.125/2, dnumc['EMDOWN']),
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
netParams.connParams['EMUP->IM'] = {
        'preConds': {'pop': 'EMUP'},
        'postConds': {'pop': 'IM'},
        #'probability': 0.125/2.,
        'convergence': prob2conv(0.125/2, dnumc['EMUP']),
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}

#Local inhibition
#I to E within area
netParams.connParams['IR->ER'] = {
        'preConds': {'pop': 'IR'},
        'postConds': {'pop': 'ER'},
        'connList': blistIRtoER,
        'weight': 0.02 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

netParams.connParams['IV1->EV1'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'EV1'},
        'connList': blistIV1toEV1,
        'weight': 0.02 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

"""
for prety in IDirPops:
  for poty in EDirPOps:
    netParams.connParams[prety+'->'+poty] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'connList': blistIV1DtoEV1D,
      'weight': 0.02 * cfg.IEGain,
      'delay': 2,
      'synMech': 'GABA', 'sec':'soma', 'loc':0.5}
"""

netParams.connParams['IV4->EV4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'EV4'},
        'connList': blistIV4toEV4,
        'weight': 0.02 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}
netParams.connParams['IMT->EMT'] = {
        'preConds': {'pop': 'IMT'},
        'postConds': {'pop': 'EMT'},
        'connList': blistIMTtoEMT,
        'weight': 0.02 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

for poty in EMotorPops: # I -> E for motor populations
  netParams.connParams['IM->'+poty] = {
    'preConds': {'pop': 'IM'},
    'postConds': {'pop': poty},
    'convergence': prob2conv(0.125, dnumc['IM']),
    'weight': 0.02 * cfg.IEGain,
    'delay': 2,
    'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

#I to I
for IType in ['IV1', 'IV4', 'IMT', 'IM']:
  netParams.connParams[IType+'->'+IType] = {
    'preConds': {'pop': IType},
    'postConds': {'pop': IType},
    'convergence': prob2conv(0.25, dnumc[IType]),
    'weight': 0.005 * cfg.IIGain, 
    'delay': 2,
    'synMech': 'GABA', 'sec':'soma', 'loc':0.5}  

#E to E feedforward connections - AMPA
netParams.connParams['ER->EV1'] = {
        'preConds': {'pop': 'ER'},
        'postConds': {'pop': 'EV1'},
        'connList': blistERtoEV1,
        'weight': 0.02 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}
netParams.connParams['EV1->EV4'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'EV4'},
        'connList': blistEV1toEV4,
        'weight': 0.01 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}
netParams.connParams['EV4->EMT'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EMT'},
        'connList': blistEV4toEMT,
        #'convergence': 10,
        'weight': 0.01 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}

"""
# these all have 0 weight, dont set them up
#E to I feedforward connections
netParams.connParams['ER->IV1'] = {
        'preConds': {'pop': 'ER'},
        'postConds': {'pop': 'IV1'},
        'connList': blistERtoIV1,
        #'convergence': 10,
        'weight': 0.00 * cfg.EIGain, #0.002
        'delay': 2,
        'synMech': 'AMPA','sec':'soma', 'loc':0.5}
netParams.connParams['EV1->IV4'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistEV1toIV4,
        #'convergence': 10,
        'weight': 0.00 * cfg.EIGain, #0.002
        'delay': 2,
        'synMech': 'AMPA','sec':'soma', 'loc':0.5}
netParams.connParams['EV4->IMT'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'IMT'},
        'connList': blistEV4toIMT,
        #'convergence': 10,
        'weight': 0.00 * cfg.EIGain, #0.002
        'delay': 2,
        'synMech': 'AMPA','sec':'soma', 'loc':0.5}
"""

"""
#E to E feedbackward connections
netParams.connParams['EV1->ER'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'ER'},
        'connList': blistEV1toER,
        #'convergence': 10,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}
netParams.connParams['EV4->EV1'] = {    # <<-- that's E -> I ?? or E -> E ?? weight is 0 but something wrong here
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EV1'},
        'connList': blistEV4toEV1,
        #'convergence': 10,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}
netParams.connParams['EMT->EV4'] = {
        'preConds': {'pop': 'EMT'},
        'postConds': {'pop': 'EV4'},
        'connList': blistEMTtoEV4,
        #'convergence': 10,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}

#I to E feedbackward connections
netParams.connParams['IV1->ER'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'ER'},
        'connList': blistIV1toER,
        #'convergence': 10,
        'weight': 0.00 * cfg.IEGain, #0.002
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}
netParams.connParams['IV4->EV1'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'EV1'},
        'connList': blistIV4toEV1,
        #'convergence': 10,
        'weight': 0.00 * cfg.IEGain, #0.002
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}
netParams.connParams['IMT->EV4'] = {
        'preConds': {'pop': 'IMT'},
        'postConds': {'pop': 'EV4'},
        'connList': blistIMTtoEV4,
        #'convergence': 10,
        'weight': 0.00 * cfg.IEGain, #0.002
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}
"""

#I to I - between areas
netParams.connParams['IV1->IV4'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistIV1toIV4,
        #'convergence': 10,
        'weight': 0.00075 * cfg.IIGain, #0.00
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}
netParams.connParams['IV4->IMT'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IMT'},
        'connList': blistIV4toIMT,
        #'convergence': 10,
        'weight': 0.00075 * cfg.IIGain, #0.00
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}

# Add connections from lower and higher visual areas to motor cortex and direct connections between premotor to motor areas
for prety in ['EV1', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW','EV1DSW', 'EV1DS','EV1DSE', 'EV4', 'EMT']:
  EEMProb = 0.1 # default
  if "EEMProb" in dconf['net']: EEMProb = dconf['net']['EEMProb']
  for poty in EMotorPops:
    for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],[dconf['net']['EEMWghtAM']*cfg.EEGain, dconf['net']['EEMWghtNM']*cfg.EEGain]):
      k = strty+prety+'->'+strty+poty
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': prob2conv(EEMProb, dnumc[prety]),
        'weight': weight,
        'delay': 2,
        'synMech': synmech,
        'sec':'dend', 'loc':0.5
      }
      if dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}

# add recurrent connectivity within EM populations
EEMRecProb = 0.0 # default
if "EEMRecProb" in dconf['net']: EEMRecProb = dconf['net']['EEMRecProb']
if EEMRecProb > 0.0:
  for ty in EMotorPops:
    for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],[dconf['net']['EEMWghtAM']*cfg.EEGain, dconf['net']['EEMWghtNM']*cfg.EEGain]):
      k = ty+ty+'->'+strty+ty
      netParams.connParams[k] = {
        'preConds': {'pop': ty},
        'postConds': {'pop': ty},
        'convergence': prob2conv(EEMRecProb, dnumc[ty]),
        'weight': weight,
        'delay': 2,
        'synMech': synmech,
        'sec':'dend', 'loc':0.5
      }
      if dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}  
        
###################################################################################################################################

sim.AIGame = None # placeholder

lsynweights = [] # list of syn weights, per node

def sumAdjustableWeightsPop (sim, popname):
  # record the plastic weights for specified popname
  lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids] # this is the set of MR cells
  W = N = 0
  for cell in lcell:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        W += float(conn['hObj'].weight[0])
        N += 1
  return W, N
  
def recordAdjustableWeightsPop (sim, t, popname):
  # record the plastic weights for specified popname
  lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids] # this is the set of MR cells
  for cell in lcell:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        lsynweights.append([t,conn.preGid,cell.gid,float(conn['hObj'].weight[0])])
  return len(lcell)
                    
def recordAdjustableWeights (sim, t, lpop = ['EMUP', 'EMDOWN']):
  """ record the STDP weights during the simulation - called in trainAgent
  """
  for pop in lpop: recordAdjustableWeightsPop(sim, t, pop)

def recordWeights (sim, t):
  """ record the STDP weights during the simulation - called in trainAgent
  """
  #lRcell = [c for c in sim.net.cells if c.gid in sim.net.pops['ER'].cellGids]
  sim.WeightsRecordingTimes.append(t)
  sim.allRLWeights.append([]) # Save this time
  sim.allNonRLWeights.append([])
  for cell in sim.net.cells:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        if conn.plast.params.RLon ==1:
          sim.allRLWeights[-1].append(float(conn['hObj'].weight[0])) # save weight only for Rl-STDP conns
        else:
          sim.allNonRLWeights[-1].append(float(conn['hObj'].weight[0])) # save weight only for nonRL-STDP conns
    
def saveWeights(sim, downSampleCells):
  ''' Save the weights for each plastic synapse '''
  with open(sim.RLweightsfilename,'w') as fid1:
    count1 = 0
    for weightdata in sim.allRLWeights:
      #fid.write('%0.0f' % weightdata[0]) # Time
      #print(len(weightdata))
      fid1.write('%0.1f' %sim.WeightsRecordingTimes[count1])
      count1 = count1+1
      for i in range(0,len(weightdata), downSampleCells): fid1.write('\t%0.8f' % weightdata[i])
      fid1.write('\n')
  print(('Saved RL weights as %s' % sim.RLweightsfilename))
  with open(sim.NonRLweightsfilename,'w') as fid2:
    count2 = 0
    for weightdata in sim.allNonRLWeights:
      #fid.write('%0.0f' % weightdata[0]) # Time
      #print(len(weightdata))
      fid2.write('%0.1f' %sim.WeightsRecordingTimes[count2])
      count2 = count2+1
      for i in range(0,len(weightdata), downSampleCells): fid2.write('\t%0.8f' % weightdata[i])
      fid2.write('\n')
  print(('Saved Non-RL weights as %s' % sim.NonRLweightsfilename))    

#    
def plotWeights():
  from pylab import figure, loadtxt, xlabel, ylabel, xlim, ylim, show, pcolor, array, colorbar
  figure()
  weightdata = loadtxt(sim.weightsfilename)
  weightdataT=list(map(list, list(zip(*weightdata))))
  vmax = max([max(row) for row in weightdata])
  vmin = min([min(row) for row in weightdata])
  pcolor(array(weightdataT), cmap='hot_r', vmin=vmin, vmax=vmax)
  xlim((0,len(weightdata)))
  ylim((0,len(weightdata[0])))
  xlabel('Time (weight updates)')
  ylabel('Synaptic connection id')
  colorbar()
  show()

def getAverageAdjustableWeights (sim, lpop = ['EMUP', 'EMDOWN']):
  # get average adjustable weights on a target population
  davg = {pop:0.0 for pop in lpop}
  for pop in lpop:
    WSum = 0; NSum = 0    
    W, N = sumAdjustableWeightsPop(sim, pop)
    # destlist_on_root = pc.py_gather(srcitem, root)
    lw = sim.pc.py_gather(W, 0)
    ln = sim.pc.py_gather(N, 0)
    if sim.rank == 0:
      WSum = W + np.sum(lw)
      NSum = N + np.sum(ln)
      #print('rank= 0, pop=',pop,'W=',W,'N=',N,'wsum=',WSum,'NSum=',NSum)
      if NSum > 0: davg[pop] = WSum / NSum    
    else:
      #destitem_from_root = sim.pc.py_scatter(srclist, root)
      pass
      #print('rank=',sim.rank,'pop=',pop,'Wm=',W,'N=',N)
  lsrc = [davg for i in range(sim.nhosts)] if sim.rank==0 else None
  dest = sim.pc.py_scatter(lsrc, 0)
  return dest

def mulAdjustableWeights (sim, dfctr):
  # multiply adjustable STDP/RL weights by dfctr[pop] value for each population keyed in dfctr
  for pop in dfctr.keys():
    if dfctr[pop] == 1.0: continue
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells
    for cell in lcell:
      for conn in cell.conns:
        if 'hSTDP' in conn:    
          conn['hObj'].weight[0] *= dfctr[pop] 

def normalizeAdjustableWeights (sim, t, lpop = ['EMUP', 'EMDOWN']):
  # normalize the STDP/RL weights during the simulation - called in trainAgent
  davg = getAverageAdjustableWeights(sim, lpop)
  try:
    dfctr = {}
    for k in lpop:
      if davg[k] < dconf['net']['EEMWghtThreshMin']:
        dfctr[k] = dconf['net']['EEMWghtThreshMin'] / davg[k]
      elif davg[k] > dconf['net']['EEMWghtThreshMax']:
        dfctr[k] = dconf['net']['EEMWghtThreshMax'] / davg[k]
      else:
        dfctr[k] = 1.0
    # normalize weights across populations to avoid bias
    if dconf['net']['EEMPopNorm']:
      mxw = np.amax([davg[k] for k in lpop])
      for k in lpop:
        if davg[k]>0.0:
          dfctr[k] *= mxw / davg[k]        
    if sim.rank==0: print('sim.rank=',sim.rank,'davg:',davg,'dfctr:',dfctr)
    mulAdjustableWeights(sim,dfctr)
  except:
    print('Exception; davg:',davg)

def saveGameBehavior(sim):
  with open(sim.ActionsRewardsfilename,'w') as fid3:
    for i in range(len(sim.allActions)):
      fid3.write('%0.1f' % sim.allTimes[i])
      fid3.write('\t%0.1f' % sim.allActions[i])
      fid3.write('\t%0.5f' % sim.allRewards[i])
      fid3.write('\t%0.1f' % sim.allProposedActions[i]) #the number of proposed action should be equal to the number of actions
      fid3.write('\t%0.1f' % sim.allHits[i]) #1 when the racket hits the ball and -1 when the racket misses the ball
      fid3.write('\n')

######################################################################################

def getFiringRatesWithInterval (trange = None, neuronal_pop = None):
  #sim.gatherData()
  spkts = sim.simData['spkt']
  spkids = sim.simData['spkid']
  pop_spikes = 0
  if len(spkts)>0:
    for i in range(len(spkids)):
      if trange[0] <= spkts[i] <= trange[1] and spkids[i] in neuronal_pop:
        pop_spikes = pop_spikes+1
    tsecs = float((trange[1]-trange[0]))/1000.0
    numCells = float(len(neuronal_pop))
    avgRates = pop_spikes/numCells/tsecs
  else:
    avgRates = 0.0
  #print('Firing rate : %.3f Hz'%(avgRates))
  return avgRates

NBsteps = 0 # this is a counter for recording the plastic weights
epCount = []
proposed_actions = [] 
total_hits = [] #numbertimes ball is hit by racket as ball changes its direction and player doesn't lose a score (assign 1). if player loses
dSTDPmech = {} # dictionary of list of STDP mechanisms
cumRewardActions = []
cumPunishingActions = []
current_time_stepNB = 0
f_ax = []
fig = []

def updateBehaviorPlot (sim,InputImages,Images,dirSensitiveNeurons,Racket_pos,Ball_pos, current_time_stepNB,f_ax,fig):
  # update 
  global cumRewardActions, cumPunishingActions
  maxtstr = len(str(100000))
  if current_time_stepNB==0:
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(4,4)
    f_ax = []
    f_ax.append(fig.add_subplot(gs[0:2,0])) #for 5-image input - 0
    f_ax.append(fig.add_subplot(gs[0:2,1])) #for single image  - 1
    f_ax.append(fig.add_subplot(gs[0:2,2])) #for direction selectivity - 2
    f_ax.append(fig.add_subplot(gs[2,0:2])) #display executed/proposed actions - 3
    f_ax.append(fig.add_subplot(gs[2,2:4])) #display - 4 
    f_ax.append(fig.add_subplot(gs[3,0:2])) #- 5
    f_ax.append(fig.add_subplot(gs[3,2:4])) #- 6
  cbaxes = fig.add_axes([0.75, 0.62, 0.01, 0.24])
  f_ax[0].cla()
  f_ax[0].imshow(InputImages[-1])
  f_ax[0].set_title('Input Images [t-5,t]')
  f_ax[2].cla()
  fa = f_ax[2].imshow(dirSensitiveNeurons,origin='upper',vmin=0, vmax=359, cmap='Dark2')
  f_ax[2].set_xlim((-0.5,9.5))
  f_ax[2].set_ylim((9.5,-0.5))
  f_ax[2].set_xticks(ticks=[0,2,4,6,8])
  f_ax[2].set_title('direction angles [t-5,t]')
  c1 = plt.colorbar(fa,cax = cbaxes)
  c1.set_ticks([22,67,112,157,202,247,292,337])
  c1.set_ticklabels(['E','NE','N','NW','W','SW','S','SE'])
  Hit_Missed = np.array(sim.allHits)
  allHit = np.where(Hit_Missed==1,1,0) 
  allMissed = np.where(Hit_Missed==-1,1,0)
  cumHits = np.cumsum(allHit) #cummulative hits evolving with time.
  cumMissHits = np.cumsum(allMissed) #if a reward is -1, replace it with 1 else replace it with 0.
  Diff_Actions_Proposed = np.subtract(sim.allActions,sim.allProposedActions)
  t0 = int(dconf['actionsPerPlay'])
  tpnts = range(t0,len(Diff_Actions_Proposed)+t0,t0)
  rewardingActions = np.sum(np.where(Diff_Actions_Proposed==0,1,0))
  punishingActions = np.sum(np.where((Diff_Actions_Proposed>0) | (Diff_Actions_Proposed<0),1,0))
  totalActs = rewardingActions + punishingActions
  cumRewardActions.append(rewardingActions/totalActs)
  cumPunishingActions.append(punishingActions/totalActs)
  f_ax[3].plot(sim.allActions,LineStyle="None",Marker=2,MarkerSize=6,MarkerFaceColor="None",MarkerEdgeColor='r')
  f_ax[3].plot(sim.allProposedActions,LineStyle="None",Marker=3,MarkerSize=6,MarkerFaceColor="None",MarkerEdgeColor='b')
  f_ax[3].set_yticks(ticks=[1,3,4])
  f_ax[3].set_yticklabels(labels=['No action','Down','Up'])
  f_ax[3].set_ylim((0.5,4.5))
  f_ax[3].legend(('Executed','Proposed'),loc='upper left')
  f_ax[4].cla()
  f_ax[4].plot(tpnts,np.array(cumRewardActions),'o-',MarkerSize=5,MarkerFaceColor='r',MarkerEdgeColor='r')
  f_ax[4].plot(tpnts,np.array(cumPunishingActions),'s-',MarkerSize=5,MarkerFaceColor='b',MarkerEdgeColor='b')
  f_ax[4].legend(('Rewarding actions','Punishing Actions'),loc='upper left')
  f_ax[5].cla()
  f_ax[5].plot(sim.allRewards,'o-',MarkerFaceColor="None",MarkerEdgeColor='g')
  f_ax[5].legend('Rewards')
  f_ax[6].cla()
  f_ax[6].plot(cumHits,Marker='o',MarkerSize=5,MarkerFaceColor='r',MarkerEdgeColor='r')
  f_ax[6].plot(cumMissHits,Marker='s',MarkerSize=3,MarkerFaceColor='k',MarkerEdgeColor='k')
  f_ax[6].legend(('Cumm. Hits','Cumm. Miss'),loc='upper left')
  f_ax[1].cla()
  for nbi in range(np.shape(Racket_pos)[0]):
    f_ax[1].imshow(Images[nbi])
    if Ball_pos[nbi][0]>18: #to account for offset for the court
      f_ax[1].plot(Racket_pos[nbi][0],Racket_pos[nbi][1],'o',MarkerSize=5, MarkerFaceColor="None",MarkerEdgeColor='r')
      f_ax[1].plot(Ball_pos[nbi][0],Ball_pos[nbi][1],'o',MarkerSize=5, MarkerFaceColor="None",MarkeredgeColor='b')
    f_ax[1].set_title('last obs')
    #plt.pause(0.1)
    ctstrl = len(str(current_time_stepNB))
    tpre = ''
    for ttt in range(maxtstr-ctstrl):
      tpre = tpre+'0'
    fn = tpre+str(current_time_stepNB)+'.png'
    fnimg = '/tmp/'+fn
    plt.savefig(fnimg)
    #plt.close() 
    #lfnimage.append(fnimg)
    current_time_stepNB = current_time_stepNB+1
  return current_time_stepNB, f_ax, fig

def updateInputRates ():
  # update the source firing rates for the ER neuron population, based on image contents
  # also update the firing rates for the direction sensitive neurons based on image contents
  lratepop = ['ER', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']  
  if sim.rank == 0:
    dFiringRates = sim.AIGame.dFiringRates    
    for k in sim.AIGame.lratepop:
      sim.pc.broadcast(sim.AIGame.dFVec[k].from_python(dFiringRates[k]),0)
      if dconf['verbose'] > 1: print('Firing Rates of',k,np.where(dFiringRates[k]==np.amax(dFiringRates[k])),np.amax(dFiringRates[k]))
  else:
    dFiringRates = OrderedDict()
    for pop in lratepop:
      vec = h.Vector()
      sim.pc.broadcast(vec,0)
      dFiringRates[pop] = vec.to_python()
      if dconf['verbose'] > 1:
        print(sim.rank,'received firing rates:',np.where(dFiringRates[pop]==np.amax(dFiringRates[pop])),np.amax(dFiringRates[pop]))          
  # update input firing rates for stimuli to R and direction sensitive cells
  for pop in lratepop:
    lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells
    offset = sim.simData['dminID'][pop]
    if dconf['verbose'] > 1: print(sim.rank,'updating len(',pop,len(lCell),'source firing rates. len(dFiringRates)=',len(dFiringRates[pop]))
    for cell in lCell:  
      for stim in cell.stims:
        if stim['source'] == 'stimMod':
          if dFiringRates[pop][int(cell.gid-offset)]==0:
            stim['hObj'].interval = 1e12
          else:  
            stim['hObj'].interval = 1000.0/dFiringRates[pop][int(cell.gid-offset)]
          #print('cell GID: ', int(cell.gid), 'vs cell ID with offset: ', int(cell.gid-R_offset)) # interval in ms as a function of rate; is cell.gid correct index??? 
      
def trainAgent (t):
  """ training interface between simulation and game environment
  """
  global NBsteps, epCount, proposed_actions, total_hits, Racket_pos, Ball_pos, current_time_stepNB, fid4
  global f_ax, fig
  global tstepPerAction
  vec = h.Vector()
  vec2 = h.Vector()
  vec3 = h.Vector()
  if t<(tstepPerAction*dconf['actionsPerPlay']): # for the first time interval use randomly selected actions
    actions =[]
    for _ in range(int(dconf['actionsPerPlay'])):
      action = dconf['movecodes'][random.randint(0,len(dconf['movecodes'])-1)]
      actions.append(action)
  else: #the actions should be based on the activity of motor cortex (MO) 1085-1093
    F_UPs = []
    F_DOWNs = []
    for ts in range(int(dconf['actionsPerPlay'])):
      ts_beg = t-tstepPerAction*(dconf['actionsPerPlay']-ts-1) 
      ts_end = t-tstepPerAction*(dconf['actionsPerPlay']-ts)
      F_UPs.append(getFiringRatesWithInterval([ts_end,ts_beg], sim.net.pops['EMUP'].cellGids))
      F_DOWNs.append(getFiringRatesWithInterval([ts_end,ts_beg], sim.net.pops['EMDOWN'].cellGids))
    sim.pc.allreduce(vec.from_python(F_UPs),1) #sum
    F_UPs = vec.to_python()
    sim.pc.allreduce(vec.from_python(F_DOWNs),1) #sum
    F_DOWNs = vec.to_python()
    if sim.rank==0:
      if fid4 is None: fid4 = open(sim.MotorOutputsfilename,'w')
      print('U,D firing rates: ', F_UPs, F_DOWNs)
      #print('Firing rates: ', F_R1, F_R2, F_R3, F_R4, F_R5, F_L1, F_L2, F_L3, F_L4, F_L5)
      fid4.write('%0.1f' % t)
      for ts in range(int(dconf['actionsPerPlay'])): fid4.write('\t%0.1f' % F_UPs[ts])
      for ts in range(int(dconf['actionsPerPlay'])): fid4.write('\t%0.1f' % F_DOWNs[ts])
      fid4.write('\n')
      actions = []
      for ts in range(int(dconf['actionsPerPlay'])):
        if F_UPs[ts]>F_DOWNs[ts]:
          actions.append(dconf['moves']['UP'])
        elif F_DOWNs[ts]>F_UPs[ts]:
          actions.append(dconf['moves']['DOWN'])
        else:
          actions.append(dconf['moves']['NOMOVE']) # No move        
  if sim.rank == 0:
    print('Model actions:', actions)
    rewards, epCount, proposed_actions, total_hits, Racket_pos, Ball_pos = sim.AIGame.playGame(actions, epCount)
    print('Proposed actions:', proposed_actions)
    if dconf['sim']['RLFakeUpRule']: # fake rule for testing reinforcing of up moves
      critic = np.sign(actions.count(dconf['moves']['UP']) - actions.count(dconf['moves']['DOWN']))          
      rewards = [critic for i in range(len(rewards))]
    elif dconf['sim']['RLFakeDownRule']: # fake rule for testing reinforcing of down moves
      critic = np.sign(actions.count(dconf['moves']['DOWN']) - actions.count(dconf['moves']['UP']))
      rewards = [critic for i in range(len(rewards))]
    elif dconf['sim']['RLFakeStayRule']: # fake rule for testing reinforcing of stay still
      critic = np.sign(actions.count(dconf['moves']['NOMOVE']) - actions.count(dconf['moves']['DOWN']) - actions.count(dconf['moves']['UP']))
      rewards = [critic for i in range(len(rewards))]                    
    else: # normal game play scoring rules
      #normal game based rewards
      critic = sum(rewards) # get critic signal (-1, 0 or 1)
      if critic>0:
        critic  = dconf['rewardcodes']['scorePoint'] 
      elif critic<0:
        critic = dconf['rewardcodes']['losePoint']  #-0.01, e.g. to reduce magnitude of punishment so rewards dominate
      else:
        critic = 0
      #starting from here not tested
      #rewards for hitting the ball
      critic_for_avoidingloss = 0
      if sum(total_hits)>0:
        critic_for_avoidingloss = dconf['rewardcodes']['hitBall'] #should be able to change this number from config file
      #rewards for following or avoiding the ball
      critic_for_following_ball = 0
      for ai in range(len(actions)):
        caction = actions[ai]
        cproposed_action = proposed_actions[ai]
        if caction - cproposed_action == 0:
          critic_for_following_ball += dconf['rewardcodes']['followBall'] #follow the ball
        else:
          critic_for_following_ball += dconf['rewardcodes']['avoidBall'] # didn't follow the ball
      #total rewards
      critic = critic + critic_for_avoidingloss + critic_for_following_ball
      rewards = [critic for i in range(len(rewards))]  # reset rewards to modified critic signal - should use more granular recording
    #till here not tested
    if dconf['verbose']:
      if critic > 0:
        print('REWARD, critic=',critic)
      elif critic < 0:
        print('PUNISH, critic=',critic)
      else:
        print('CRITIC=0')                    
    sim.pc.broadcast(vec.from_python([critic]), 0) # convert python list to hoc vector to broadcast critic value to other nodes
    UPactions = np.sum(np.where(np.array(actions)==dconf['moves']['UP'],1,0))
    DOWNactions = np.sum(np.where(np.array(actions)==dconf['moves']['DOWN'],1,0))
    sim.pc.broadcast(vec2.from_python([UPactions]),0)
    sim.pc.broadcast(vec3.from_python([DOWNactions]),0)
  else: # other workers
    sim.pc.broadcast(vec, 0) # receive critic value from master node
    critic = vec.to_python()[0] # critic is first element of the array
    sim.pc.broadcast(vec2, 0)
    UPactions = vec2.to_python()[0]
    sim.pc.broadcast(vec3, 0)
    DOWNactions = vec3.to_python()[0]
    if dconf['verbose']: print('UPactions: ', UPactions,'DOWNactions: ', DOWNactions)
  if critic != 0: # if critic signal indicates punishment (-1) or reward (+1)
    if sim.rank==0: print('t=',t,'- adjusting weights based on RL critic value:', critic)
    if not dconf['sim']['targettedRL'] or UPactions==DOWNactions:
      if dconf['verbose']: print('APPLY RL to both EMUP and EMDOWN')
      for STDPmech in dSTDPmech['all']: STDPmech.reward_punish(float(critic))
    elif UPactions>DOWNactions:
      if dconf['verbose']: print('APPLY RL to EMUP')
      for STDPmech in dSTDPmech['EMUP']: STDPmech.reward_punish(float(critic))
    elif DOWNactions>UPactions:
      if dconf['verbose']: print('APPLY RL to EMDOWN')
      for STDPmech in dSTDPmech['EMDOWN']: STDPmech.reward_punish(float(critic))
  if sim.rank==0:
    print('Game rewards:', rewards) # only rank 0 has access to rewards      
    for action in actions:
        sim.allActions.append(action)
    for pactions in proposed_actions: #also record proposed actions
        sim.allProposedActions.append(pactions)
    for reward in rewards: # this generates an error - since rewards only declared for sim.rank==0; bug?
        sim.allRewards.append(reward)
    for hits in total_hits:
        sim.allHits.append(hits) #hit or no hit
    tvec_actions = []
    for ts in range(len(actions)): tvec_actions.append(t-tstepPerAction*(len(actions)-ts-1))
    for ltpnt in tvec_actions: sim.allTimes.append(ltpnt)
    #current_time_stepNB, f_ax, fig = updateBehaviorPlot (sim,sim.AIGame.ReducedImages,sim.AIGame.FullImages,dirSensitiveNeurons,Racket_pos,Ball_pos,current_time_stepNB, f_ax, fig)
    #current_time_stepNB = current_time_stepNB + 1
  updateInputRates() # update firing rate of inputs to R population (based on image content)                
  NBsteps += 1
  if NBsteps % recordWeightStepSize == 0:
    if dconf['verbose'] > 0 and sim.rank==0:
      print('Weights Recording Time:', t, 'NBsteps:',NBsteps,'recordWeightStepSize:',recordWeightStepSize)
    recordAdjustableWeights(sim, t) 
    #recordWeights(sim, t)
  if NBsteps % normalizeWeightStepSize == 0:
    if dconf['verbose'] > 0 and sim.rank==0:
      print('Weight Normalize Time:', t, 'NBsteps:',NBsteps,'normalizeWeightStepSize:',normalizeWeightStepSize)
    normalizeAdjustableWeights(sim, t)     

def getAllSTDPObjects (sim):
  # get all the STDP objects from the simulation's cells
  dSTDPmech = {'all':[], 'EMUP':[], 'EMDOWN':[]} # dictionary of STDP objects keyed by type (all, for EMUP, EMDOWN populations)
  for cell in sim.net.cells:
    for conn in cell.conns:
      STDPmech = conn.get('hSTDP')  # check if has STDP mechanism
      if STDPmech:
        dSTDPmech['all'].append(STDPmech)
        for pop in ['EMUP', 'EMDOWN']:
          if cell.gid in sim.net.pops[pop].cellGids:
            dSTDPmech[pop].append(STDPmech)
  return dSTDPmech
        
#Alterate to create network and run simulation
# create network object and set cfg and net params; pass simulation config and network params as arguments
sim.initialize(simConfig = simConfig, netParams = netParams)
sim.net.createPops()                      # instantiate network populations
sim.net.createCells()                     # instantiate network cells based on defined populations
sim.net.connectCells()                    # create connections between cells based on params
sim.net.addStims()                      #instantiate netStim
sim.setupRecording()                  # setup variables to record for each cell (spikes, V traces, etc)

dSTDPmech = getAllSTDPObjects(sim) # get all the STDP objects up-front

def updateSTDPWeights (sim, W):
  #this function assign weights stored in 'ResumeSimFromFile' to all connections by matching pre and post neuron ids  
  # get all the simulation's cells (on a given node)
  for cell in sim.net.cells:
    cpostID = cell.gid#find postID
    for conn in cell.conns:
      if not 'hSTDP' in conn: continue
      cpreID = conn.preGid  #find preID
      if type(cpreID) != int: continue
      cConnW = W[(W.postid==cpostID) & (W.preid==cpreID)] #find the record for a connection with pre and post neuron ID
      #find weight for the STDP connection between preID and postID
      for idx in cConnW.index:
        cW = cConnW.at[idx,'weight']
        conn['hObj'].weight[0] = cW
        if dconf['verbose'] > 1: print('weight updated:', cW)

#if specified 'ResumeSim' = 1, load the connection data from 'ResumeSimFromFile' and assign weights to STDP synapses  
if dconf['simtype']['ResumeSim']:
  try:
    from simdat import readweightsfile2pdf
    A = readweightsfile2pdf(dconf['simtype']['ResumeSimFromFile'])
    updateSTDPWeights(sim, A[A.time == max(A.time)]) # take the latest weights saved
    if sim.rank==0: print('Updated STDP weights')
  except:
    print('Could not restore STDP weights from file.')

if sim.rank == 0: 
  from aigame import AIGame
  sim.AIGame = AIGame() # only create AIGame on node 0
  # node 0 saves the json config file
  # this is just a precaution since simConfig pkl file has MOST of the info; ideally should adjust simConfig to contain
  # ALL of the required info
  from utils import backupcfg
  backupcfg(dconf['sim']['name'])

def setdminID (sim, lpop):
  # setup min ID for each population in lpop
  alltags = sim._gatherAllCellTags() #gather cell tags; see https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/sim/gather.py
  dGIDs = {pop:[] for pop in lpop}
  for tinds in range(len(alltags)):
    if alltags[tinds]['pop'] in lpop:
      dGIDs[alltags[tinds]['pop']].append(tinds)
  sim.simData['dminID'] = {pop:np.amin(dGIDs[pop]) for pop in lpop}

setdminID(sim, ['ER', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE'])

tPerPlay = tstepPerAction*dconf['actionsPerPlay']
sim.runSimWithIntervalFunc(tPerPlay,trainAgent) # has periodic callback to adjust STDP weights based on RL signal
if sim.rank==0 and fid4 is not None: fid4.close()
sim.gatherData() # gather data from different nodes
sim.saveData() # save data to disk

def LSynWeightToD (L):
  # convert list of synaptic weights to dictionary to save disk space
  print('converting synaptic weight list to dictionary...')
  dout = {}
  for row in L:
    t,preID,poID,w = row
    if preID not in dout: dout[preID] = {}
    if poID not in dout[preID]: dout[preID][poID] = []
    dout[preID][poID].append([t,w])
  return dout

def saveSynWeights ():
  # save synaptic weights 
  fn = 'data/'+dconf['sim']['name']+'synWeights_'+str(sim.rank)+'.pkl'
  pickle.dump(lsynweights, open(fn, 'wb')) # save synaptic weights to disk for this node
  sim.pc.barrier() # wait for other nodes
  time.sleep(1)    
  if sim.rank == 0: # rank 0 reads and assembles the synaptic weights into a single output file
    L = []
    for i in range(sim.nhosts):
      fn = 'data/'+dconf['sim']['name']+'synWeights_'+str(i)+'.pkl'
      while not os.path.isfile(fn): # wait until the file is written/available
        print('saveSynWeights: waiting for finish write of', fn)
        time.sleep(1)      
      lw = pickle.load(open(fn,'rb'))
      print(fn,'len(lw)=',len(lw),type(lw),type(lw[0]),lw[0])
      os.unlink(fn) # remove the temporary file
      L = L + lw # concatenate to the list L
    #pickle.dump(L,open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb')) # this would save as a List
    # now convert the list to a dictionary to save space, and save it to disk
    pickle.dump(LSynWeightToD(L),open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb'))    

if sim.saveWeights: saveSynWeights()

def saveMotionFields (ldflow): pickle.dump(ldflow, open('data/'+dconf['sim']['name']+'MotionFields.pkl', 'wb'))

def saveInputImages (Images):
  # save input images to txt file (switch to pkl?)
  InputImages = np.array(Images)
  print(InputImages.shape)  
  with open('data/'+dconf['sim']['name']+'InputImages.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(InputImages.shape))
    for Input_Image in InputImages:
      np.savetxt(outfile, Input_Image, fmt='%-7.2f')
      outfile.write('# New slice\n')
      
if sim.rank == 0: # only rank 0 should save. otherwise all the other nodes could over-write the output or quit first; rank 0 plots
  if dconf['sim']['doplot']:
    print('plot raster:')
    sim.analysis.plotData()    
  if sim.plotWeights: plotWeights() 
  saveGameBehavior(sim)
  fid5 = open('data/'+dconf['sim']['name']+'ActionsPerEpisode.txt','w')
  for i in range(len(epCount)):
    fid5.write('\t%0.1f' % epCount[i])
    fid5.write('\n')
  if sim.saveInputImages: saveInputImages(sim.AIGame.ReducedImages)
  #anim.savemp4('/tmp/*.png','data/'+dconf['sim']['name']+'randGameBehavior.mp4',10)
  if sim.saveMotionFields: saveMotionFields(sim.AIGame.ldflow)
  if dconf['sim']['doquit']: quit()
