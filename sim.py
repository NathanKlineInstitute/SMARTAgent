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
sim.topologicalConns = dict() # dictionary to save topological connections.
#sim.NonRLweightsfilename = 'data/'+dconf['sim']['name']+'NonRLweights.txt'  # file to store weights
sim.plotWeights = 0  # plot weights
sim.saveWeights = 1  # save weights
if 'saveWeights' in dconf['sim']: sim.saveWeights = dconf['sim']['saveWeights']
sim.saveInputImages = 1 #save Input Images (5 game frames)
sim.saveMotionFields = 1 # whether to save the motion fields
sim.saveObjPos = 1 # save ball and paddle position to file
sim.saveAssignedFiringRates = dconf['sim']['saveAssignedFiringRates']
recordWeightStepSize = dconf['sim']['recordWeightStepSize']
normalizeWeightStepSize = dconf['sim']['normalizeWeightStepSize']
#recordWeightDT = 1000 # interval for recording synaptic weights (change later)
recordWeightDCells = 1 # to record weights for sub samples of neurons
tstepPerAction = dconf['sim']['tstepPerAction'] # time step per action (in ms)

fid4=None # only used by rank 0

scale = dconf['net']['scale'] # scales the size of the network (only number of neurons)

ETypes = dconf['net']['ETypes'] # excitatory neuron types
ITypes = dconf['net']['ITypes'] # inhibitory neuron types
allpops = list(dconf['net']['allpops'].keys())
EMotorPops = dconf['net']['EMotorPops'] # excitatory neuron motor populations
EPreAPops = dconf['net']['EPreAPops'] # excitatory premotor populations
EPreADirPops = dconf['net']['EPreADirPops']
EVPops = [pop for pop in EPreAPops if pop not in EPreADirPops] # excitatory visual populations (non direction selective)

dnumc = OrderedDict({ty:dconf['net']['allpops'][ty]*scale for ty in allpops}) # number of neurons of a given type

def getpadding ():
  # get padding-related parameters
  dnumc_padx = OrderedDict({ty:dconf['net']['allpops'][ty]*0 for ty in allpops}) # a dictionary with zeros to keep number of padded neurons in one dimension
  dtopoldivcons = dconf['net']['alltopoldivcons']
  dtopolconvcons = dconf['net']['alltopolconvcons']
  allpops_withconvtopology = list(dtopolconvcons.keys())
  allpops_withdivtopology = list(dtopoldivcons.keys())
  # below is the code for updating neuronal pop size to include padding. 
  if dconf['net']['useNeuronPad']: # PADDING NEEDS TO BE FIXED..... DONT USE IT UNTIL FIXED
    # first make dicionary of paddings in each dimension for each pop
    for pop in allpops_withconvtopology:
      receptive_fields = []
      for postpop in list(dtopolconvcons[pop].keys()):
        if dnumc[postpop]>0:
          receptive_fields.append(dtopolconvcons[pop][postpop])
      if len(receptive_fields)>0:
        max_receptive_field = np.amax(receptive_fields)
      else:
        max_receptive_field = 0
      if dnumc[pop]>0 and max_receptive_field>0:
        dnumc_padx[pop] = max_receptive_field-1
    for pop in allpops_withdivtopology:
      receptive_fields = []
      for postpop in list(dtopoldivcons[pop].keys()):
        if dnumc[postpop]>0:
          receptive_fields.append(dtopoldivcons[pop][postpop])
      if len(receptive_fields)>0:
        max_receptive_field = np.amax(receptive_fields)
      else:
        max_receptive_field = 0
      if dnumc[pop]>0 and max_receptive_field>0:
        dnumc_padx[pop] = max_receptive_field-1
    for pop in allpops:
      if dnumc[pop]>0 and dnumc_padx[pop]>0:
        dnumc[pop] = int((np.sqrt(dnumc[pop])+dnumc_padx[pop])**2)
    dnumc_padx['EMUP'] = 2
    dnumc_padx['EMDOWN'] = 2
    if dnumc['EMSTAY']>0: dnumc_padx['EMSTAY'] = 2
    dnumc['EMUP'] = int((np.sqrt(dnumc['EMUP'])+dnumc_padx['EMUP'])**2)
    dnumc['EMDOWN'] = int((np.sqrt(dnumc['EMDOWN'])+dnumc_padx['EMDOWN'])**2)
  return dnumc_padx, dtopoldivcons,dtopolconvcons,allpops_withconvtopology,allpops_withdivtopology

dnumc_padx, dtopoldivcons,dtopolconvcons,allpops_withconvtopology,allpops_withdivtopology = getpadding()
  
if dnumc['EMSTAY']>0:
  lrecpop = ['EMUP', 'EMDOWN','EMSTAY','EA'] # which populations to record from
else:
  lrecpop = ['EMUP', 'EMDOWN','EA'] # which populations to record from
  
if dconf['net']['EEPreAProb'] > 0.0 or dconf['net']['EEMFeedbackProb'] > 0.0 or dconf['net']['VisualFeedback']:
  for pop in ['EV1','EV1DE','EV1DNE','EV1DN','EV1DNW','EV1DW','EV1DSW','EV1DS','EV1DSE','EV4','EMT']:
    lrecpop.append(pop)
  if dconf['net']['VisualFeedback'] and dnumc['ER']>0: lrecpop.append('ER')

if dconf['net']['RLconns']['Visual'] or dconf['net']['STDPconns']['Visual']:
  if lrecpop.count('EV4')==0: lrecpop.append('EV4')
  if lrecpop.count('EMT')==0: lrecpop.append('EMT')
  
if dconf['net']['RLconns']['EIPlast'] or dconf['net']['STDPconns']['EIPlast']:
  lrecpop.append('IM')
  #lrecpop.append('ID')
  lrecpop.append('IMUP')
  lrecpop.append('IMDOWN')
  lrecpop.append('IMSTAY')

# Network parameters
netParams = specs.NetParams() #object of class NetParams to store the network parameters
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells

simConfig = specs.SimConfig()           # object of class SimConfig to store simulation configuration
#Simulation options
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
#simConfig.createNEURONObj = True  # create HOC objects when instantiating network
#simConfig.createPyStruct = True  # create Python structure (simulator-independent) when instantiating network
simConfig.analysis['plotTraces'] = {'include': [(pop, 0) for pop in ['ER','IR','EV1','EV1DE','ID','IV1','EV4','IV4','EMT','IMT','EMDOWN','EMUP','EMSTAY','IM','IMUP','IMDOWN','EA','IA']]}
simConfig.analysis['plotRaster'] = {'popRates':'overlay','showFig':dconf['sim']['doplot']}
#simConfig.analysis['plot2Dnet'] = True 
#simConfig.analysis['plotConn'] = True           # plot connectivity matrix

ECellModel = dconf['net']['ECellModel']
ICellModel = dconf['net']['ICellModel']

#Population parameters
for ty in allpops:
  if ty in ETypes:
    netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': ECellModel}
  else:
    netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': ICellModel}

def makeECellModel (ECellModel):
  # create rules for excitatory neuron models
  EExcitSec = 'dend' # section where excitatory synapses placed  
  if ECellModel == 'Mainen':    
    netParams.importCellParams(label='PYR_Mainen_rule', conds={'cellType': ETypes}, fileName='cells/mainen.py', cellName='PYR2')
    netParams.cellParams['PYR_Mainen_rule']['secs']['soma']['threshold'] = 0.0
    EExcitSec = 'dend' # section where excitatory synapses placed
  elif ECellModel == 'IzhiRS':   ## RS Izhi cell params
    EExcitSec = 'soma' # section where excitatory synapses placed
    RScellRule = {'conds': {'cellType': ETypes, 'cellModel': 'IzhiRS'}, 'secs': {}}
    RScellRule['secs']['soma'] = {'geom': {}, 'pointps':{}}  #  soma
    RScellRule['secs']['soma']['geom'] = {'diam': 10, 'L': 10, 'cm': 31.831}
    RScellRule['secs']['soma']['pointps']['Izhi'] = {
      'mod':'Izhi2007b', 'C':1, 'k':0.7, 'vr':-60, 'vt':-40, 'vpeak':35, 'a':0.03, 'b':-2, 'c':-50, 'd':100, 'celltype':1
    }
    netParams.cellParams['IzhiRS'] = RScellRule  # add dict to list of cell properties
  elif ECellModel == 'IntFire4':
    EExcitSec = 'soma' # section where excitatory synapses placed
    simConfig.recordTraces = {'V_soma':{'var':'m'}}  # Dict with traces to record
    for ty in ETypes:
      netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': ECellModel}#, 'params':{'taue':5.35,'taui1':9.1,'taui2':0.07,'taum':20}}    
  elif ECellModel == 'Friesen':
    cellRule = netParams.importCellParams(label='PYR_Friesen_rule', conds={'cellType': ETypes, 'cellModel': 'Friesen'},
                fileName='cells/friesen.py', cellName='MakeRSFCELL')
    cellRule['secs']['axon']['spikeGenLoc'] = 0.5  # spike generator location.
    EExcitSec = 'dend' # section where excitatory synapses placed
  return EExcitSec

def makeICellModel (ICellModel):
  # create rules for inhibitory neuron models
  if ICellModel == 'FS_BasketCell':    ## FS Izhi cell params
    netParams.importCellParams(label='FS_BasketCell_rule', conds={'cellType': ITypes}, fileName='cells/FS_BasketCell.py', cellName='Bas')
    netParams.cellParams['FS_BasketCell_rule']['secs']['soma']['threshold'] = -10.0
  elif ICellModel == 'IzhiFS': # defaults to Izhi cell otherwise
    FScellRule = {'conds': {'cellType': ITypes, 'cellModel': 'IzhiFS'}, 'secs': {}}
    FScellRule['secs']['soma'] = {'geom': {}, 'pointps':{}}  #  soma
    FScellRule['secs']['soma']['geom'] = {'diam': 10, 'L': 10, 'cm': 31.831}
    FScellRule['secs']['soma']['pointps']['Izhi'] = {
      'mod':'Izhi2007b', 'C':0.2, 'k':1.0, 'vr':-55, 'vt':-40, 'vpeak':25, 'a':0.2, 'b':-2, 'c':-45, 'd':-55, 'celltype':5
    }
    netParams.cellParams['IzhiFS'] = FScellRule  # add dict to list of cell properties
  elif ICellModel == 'IntFire4':
    simConfig.recordTraces = {'V_soma':{'var':'m'}}  # Dict with traces to record
    for ty in ITypes:
      netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': ICellModel}#, 'params':{'taue':5.35,'taui1':9.1,'taui2':0.07,'taum':20}}        
  elif ICellModel == 'Friesen':
    cellRule = netParams.importCellParams(label='Bas_Friesen_rule', conds={'cellType': ITypes, 'cellModel': 'Friesen'},
                fileName='cells/friesen.py', cellName='MakeFSFCELL')
    cellRule['secs']['axon']['spikeGenLoc'] = 0.5  # spike generator location.  
      
EExcitSec = makeECellModel(ECellModel)
makeICellModel(ICellModel)
  
## Synaptic mechanism parameters
netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}  # excitatory synaptic mechanism
netParams.synMechParams['NMDA'] = {'mod': 'Exp2Syn', 'tau1': 0.15, 'tau2': 166.0, 'e': 0} # NMDA
netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80}  # inhibitory synaptic mechanism

#wmin should be set to the initial/baseline weight of the connection.
STDPparams = {'hebbwt': 0.0001, 'antiwt':-0.00001, 'wbase': 0.0012, 'wmax': 50, 'RLon': 0 , 'RLhebbwt': 0.001, 'RLantiwt': -0.000,
              'tauhebb': 10, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 0, 'verbose':0}


dSTDPparamsRL = {} # STDP-RL parameters for AMPA,NMDA synapses; generally uses shorter/longer eligibility traces
for sy in ['AMPA', 'NMDA']: dSTDPparamsRL[sy] = dconf['RL'][sy]
if 'AMPAI' in dconf['RL']: dSTDPparamsRL['AMPAI'] = dconf['RL']['AMPAI']

dSTDPparams = {} # STDPL parameters for AMPA,NMDA synapses; generally uses shorter/longer eligibility traces
for sy in ['AMPA', 'NMDA']: dSTDPparams[sy] = dconf['STDP'][sy]
if 'AMPAI' in dconf['STDP']: dSTDPparams['AMPAI'] = dconf['STDP']['AMPAI']

# these are the image-based inputs provided to the R (retinal) cells
netParams.stimSourceParams['stimMod'] = {'type': 'NetStim', 'rate': 'variable', 'noise': 0}

stimModInputW = dconf['net']['stimModInputW']
stimModDirW = dconf['net']['stimModDirW']

inputPop = 'EV1' # which population gets the direct visual inputs (pixels)
if dnumc['ER']>0: inputPop = 'ER'
netParams.stimTargetParams['stimMod->'+inputPop] = {
  'source': 'stimMod',
  'conds': {'pop': inputPop},
  'convergence': 1,
  'weight': stimModInputW,
  'delay': 1,
  'synMech': 'AMPA'
}

for pop in ['EV1D'+Dir for Dir in ['E','NE','N', 'NW','W','SW','S','SE']]:
  netParams.stimTargetParams['stimMod->'+pop] = {
    'source': 'stimMod',
    'conds': {'pop': pop},
    'convergence': 1,
    'weight': stimModDirW,
    'delay': 1,
    'synMech': 'AMPA'
  }

#background input to inhibitory neurons to increase their firing rate

# Stimulation parameters

""" 
# weights are currently at 0 so do not need to simulate them
netParams.stimSourceParams['ebkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['ebkg->all'] = {
'source': 'ebkg', 'conds': {'cellType': ['EV1','EV4','EMT']}, 'weight': 0.0, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'
}

netParams.stimSourceParams['MLbkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['MLbkg->all'] = {'source': 'MLbkg', 'conds': {'cellType': ['EMDOWN']}, 'weight': 0.0, 'delay': 1, 'synMech': 'AMPA'}

netParams.stimSourceParams['MRbkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['MRbkg->all'] = {'source': 'MRbkg', 'conds': {'cellType': ['EMUP']}, 'weight': 0.0, 'delay': 1, 'synMech': 'AMPA'}

netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 20, 'noise': 1.0}
netParams.stimTargetParams['bkg->all'] = {
'source': 'bkg', 'conds': {'cellType': ['IR','IV1','IV4','IMT']}, 'weight': 0.0, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'
}
"""
# setup noise inputs
for ty,sy in zip(["E","I"],["AMPA","GABA"]):
  Weight,Rate = dconf["Noise"][ty]["Weight"],dconf["Noise"][ty]["Rate"]
  if Weight > 0.0 and Rate > 0.0: # only create the netstims if rate,weight > 0
    netParams.stimSourceParams[ty+'Mbkg'] = {'type': 'NetStim', 'rate': Rate, 'noise': 1.0}
    netParams.stimTargetParams[ty+'Mbkg->all'] = {
      'source': ty+'Mbkg', 'conds': {'cellType': EMotorPops}, 'weight': Weight, 'delay': 'max(1, normal(5,2))', 'synMech': sy
    }
      
######################################################################################

#####################################################################################
#Feedforward excitation
#E to E - Feedforward connections
if dconf['sim']['useReducedNetwork']:
  if dconf['sim']['captureTwoObjs']:
    cLV1toEA, cLV1DEtoEA, cLV1DNEtoEA, cLV1DNtoEA, cLV1DNWtoEA, cLV1DWtoEA, cLV1DSWtoEA, cLV1DStoEA, cLV1DSEtoEA = createConnListV1toEA2(dnumc['EV1'],2) # 3 objects in the game
  else:
    cLV1toEA, cLV1DEtoEA, cLV1DNEtoEA, cLV1DNtoEA, cLV1DNWtoEA, cLV1DWtoEA, cLV1DSWtoEA, cLV1DStoEA, cLV1DSEtoEA = createConnListV1toEA(dnumc['EV1'],3) # 3 objects in the game
  #cLV1toEA, cLV1DEtoEA, cLV1DNEtoEA, cLV1DNtoEA, cLV1DNWtoEA, cLV1DWtoEA, cLV1DSWtoEA, cLV1DStoEA, cLV1DSEtoEA = createConnListV1toEA(60,3)
else:
  if dnumc['ER']>0: blistERtoEV1, connCoordsERtoEV1 = connectLayerswithOverlap(NBpreN = dnumc['ER'], NBpostN = dnumc['EV1'], overlap_xdir = dtopolconvcons['ER']['EV1'], padded_preneurons_xdir = dnumc_padx['ER'], padded_postneurons_xdir = dnumc_padx['EV1'])
  blistEV1toEV4, connCoordsEV1toEV4 = connectLayerswithOverlap(NBpreN = dnumc['EV1'], NBpostN = dnumc['EV4'], overlap_xdir = dtopolconvcons['EV1']['EV4'], padded_preneurons_xdir = dnumc_padx['EV1'], padded_postneurons_xdir = dnumc_padx['EV4'])
  blistEV4toEMT, connCoordsEV4toEMT = connectLayerswithOverlap(NBpreN = dnumc['EV4'], NBpostN = dnumc['EMT'], overlap_xdir = dtopolconvcons['EV4']['EMT'], padded_preneurons_xdir = dnumc_padx['EV4'], padded_postneurons_xdir = dnumc_padx['EMT'])
  #E to I - WithinLayer connections
  if dnumc['ER']>0: blistERtoIR, connCoordsERtoIR = connectLayerswithOverlap(NBpreN = dnumc['ER'], NBpostN = dnumc['IR'], overlap_xdir = dtopolconvcons['ER']['IR'], padded_preneurons_xdir = dnumc_padx['ER'], padded_postneurons_xdir = dnumc_padx['IR'])
  blistEV1toIV1, connCoordsEV1toIV1 = connectLayerswithOverlap(NBpreN = dnumc['EV1'], NBpostN = dnumc['IV1'], overlap_xdir = dtopolconvcons['EV1']['IV1'], padded_preneurons_xdir = dnumc_padx['EV1'], padded_postneurons_xdir = dnumc_padx['IV1'])
  blistEV4toIV4, connCoordsEV4toIV4 = connectLayerswithOverlap(NBpreN = dnumc['EV4'], NBpostN = dnumc['IV4'], overlap_xdir = dtopolconvcons['EV4']['IV4'], padded_preneurons_xdir = dnumc_padx['EV4'], padded_postneurons_xdir = dnumc_padx['IV4'])
  blistEMTtoIMT, connCoordsEMTtoIMT = connectLayerswithOverlap(NBpreN = dnumc['EMT'], NBpostN = dnumc['IMT'], overlap_xdir = dtopolconvcons['EMT']['IMT'], padded_preneurons_xdir = dnumc_padx['EMT'], padded_postneurons_xdir = dnumc_padx['IMT'])
  #I to E - WithinLayer Inhibition
  if dnumc['IR']>0: blistIRtoER, connCoordsIRtoER = connectLayerswithOverlapDiv(NBpreN = dnumc['IR'], NBpostN = dnumc['ER'], overlap_xdir = dtopoldivcons['IR']['ER'], padded_preneurons_xdir = dnumc_padx['IR'], padded_postneurons_xdir = dnumc_padx['ER'])
  blistIV1toEV1, connCoordsIV1toEV1 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV1'], NBpostN = dnumc['EV1'], overlap_xdir = dtopoldivcons['IV1']['EV1'], padded_preneurons_xdir = dnumc_padx['IV1'], padded_postneurons_xdir = dnumc_padx['EV1'])
  blistIV4toEV4, connCoordsIV4toEV4 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV4'], NBpostN = dnumc['EV4'], overlap_xdir = dtopoldivcons['IV4']['EV4'], padded_preneurons_xdir = dnumc_padx['IV4'], padded_postneurons_xdir = dnumc_padx['EV4'])
  blistIMTtoEMT, connCoordsIMTtoEMT = connectLayerswithOverlapDiv(NBpreN = dnumc['IMT'], NBpostN = dnumc['EMT'], overlap_xdir = dtopoldivcons['IMT']['EMT'], padded_preneurons_xdir = dnumc_padx['IMT'], padded_postneurons_xdir = dnumc_padx['EMT'])
  #Feedbackward excitation
  #E to E 
  if dnumc['ER']>0: blistEV1toER, connCoordsEV1toER = connectLayerswithOverlapDiv(NBpreN = dnumc['EV1'], NBpostN = dnumc['ER'], overlap_xdir = dtopoldivcons['EV1']['ER'], padded_preneurons_xdir = dnumc_padx['EV1'], padded_postneurons_xdir = dnumc_padx['ER'])
  blistEV4toEV1, connCoordsEV4toEV1  = connectLayerswithOverlapDiv(NBpreN = dnumc['EV4'], NBpostN = dnumc['EV1'], overlap_xdir = dtopoldivcons['EV4']['EV1'], padded_preneurons_xdir = dnumc_padx['EV4'], padded_postneurons_xdir = dnumc_padx['EV1'])
  blistEMTtoEV4, connCoordsEMTtoEV4  = connectLayerswithOverlapDiv(NBpreN = dnumc['EMT'], NBpostN = dnumc['EV4'], overlap_xdir = dtopoldivcons['EMT']['EV4'], padded_preneurons_xdir = dnumc_padx['EMT'], padded_postneurons_xdir = dnumc_padx['EV4'])
  #Feedforward inhibition
  #I to I
  blistIV1toIV4, connCoordsIV1toIV4 = connectLayerswithOverlap(NBpreN = dnumc['IV1'], NBpostN = dnumc['IV4'], overlap_xdir = dtopolconvcons['IV1']['IV4'], padded_preneurons_xdir = dnumc_padx['IV1'], padded_postneurons_xdir = dnumc_padx['IV4'])
  blistIV4toIMT, connCoordsIV4toIMT = connectLayerswithOverlap(NBpreN = dnumc['IV4'], NBpostN = dnumc['IMT'], overlap_xdir = dtopolconvcons['IV4']['IMT'], padded_preneurons_xdir = dnumc_padx['IV4'], padded_postneurons_xdir = dnumc_padx['IMT'])
  #Feedbackward inhibition
  #I to E 
  if dnumc['IR']>0: blistIV1toER, connCoordsIV1toER = connectLayerswithOverlapDiv(NBpreN = dnumc['IV1'], NBpostN = dnumc['ER'], overlap_xdir = dtopoldivcons['IV1']['ER'], padded_preneurons_xdir = dnumc_padx['IV1'], padded_postneurons_xdir = dnumc_padx['ER'])
  blistIV4toEV1, connCoordsIV4toEV1 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV4'], NBpostN = dnumc['EV1'], overlap_xdir = dtopoldivcons['IV4']['EV1'], padded_preneurons_xdir = dnumc_padx['IV4'], padded_postneurons_xdir = dnumc_padx['EV1'])
  blistIMTtoEV4, connCoordsIMTtoEV4 = connectLayerswithOverlapDiv(NBpreN = dnumc['IMT'], NBpostN = dnumc['EV4'], overlap_xdir = dtopoldivcons['IMT']['EV4'], padded_preneurons_xdir = dnumc_padx['IMT'], padded_postneurons_xdir = dnumc_padx['EV4'])
  
# synaptic weight gain (based on E, I types)
cfg = simConfig
cfg.EEGain = dconf['net']['EEGain'] # E to E scaling factor
cfg.EIGain = dconf['net']['EIGain'] # E to I scaling factor
cfg.IEGain = dconf['net']['IEGain'] # I to E scaling factor
cfg.IIGain = dconf['net']['IIGain'] # I to I scaling factor

### from https://www.neuron.yale.edu/phpBB/viewtopic.php?f=45&t=3770&p=16227&hilit=memory#p16122
cfg.saveCellSecs = bool(dconf['sim']['saveCellSecs']) # if False removes all data on cell sections prior to gathering from nodes
cfg.saveCellConns = bool(dconf['sim']['saveCellConns']) # if False removes all data on cell connections prior to gathering from nodes
###

# weight variance -- check if need to vary the initial weights (note, they're over-written if resumeSim==1)
cfg.weightVar = dconf['net']['weightVar']

def getInitWeight (weight):
  """get initial weight for a connection
     checks if weightVar is non-zero, if so will use a uniform distribution 
     with range on interval: (1-var)*weight, (1+var)*weight
  """
  if cfg.weightVar == 0.0:
    return weight
  else:
    print('uniform(%g,%g)' % (weight*(1.0-cfg.weightVar),weight*(1.0+cfg.weightVar)))
    return 'uniform(%g,%g)' % (weight*(1.0-cfg.weightVar),weight*(1.0+cfg.weightVar))

#Local excitation
#E to E recurrent connectivity in preassociation areas
EEPreAProb = dconf['net']['EEPreAProb']
if EEPreAProb > 0.0:
  for epop in EPreAPops:
    if dnumc[epop] <= 0: continue # skip rule setup for empty population
    prety = poty = epop
    for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],[dconf['net']['EEPreAWghtAM']*cfg.EEGain, dconf['net']['EEPreAWghtNM']*cfg.EEGain]):
      k = strty+prety+'->'+strty+poty
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': prob2conv(EEPreAProb, dnumc[prety]),
        'weight': getInitWeight(weight),
        'delay': 2,
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5
      }
      useRL = useSTDP = False
      if prety in dconf['net']['EPreADirPops']:
        if dconf['net']['RLconns']['RecurrentDirNeurons']: useRL = True
        if dconf['net']['STDPconns']['RecurrentDirNeurons']: useSTDP = True
      if prety in dconf['net']['EPreALocPops']:
        if dconf['net']['RLconns']['RecurrentLocNeurons']: useRL = True
        if dconf['net']['STDPconns']['RecurrentLocNeurons']: useSTDP = True        
      if useRL and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif useSTDP and dSTDPparams[synmech]['STDPon']:
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}                

VTopoI = dconf['net']['VTopoI'] # whether visual neurons have topological arrangement
        
#E to I within area
if dnumc['ER']>0:
  netParams.connParams['ER->IR'] = {
          'preConds': {'pop': 'ER'},
          'postConds': {'pop': 'IR'},
          'weight': 0.02 * cfg.EIGain,
          'delay': 2,
          'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
  if VTopoI and dconf['sim']['useReducedNetwork']==0: netParams.connParams['ER->IR']['connList'] = blistERtoIR
  else: netParams.connParams['ER->IR']['convergence'] = prob2conv(0.0225, dnumc['EV1'])
  
netParams.connParams['EV1->IV1'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'IV1'},
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}

if VTopoI and dconf['sim']['useReducedNetwork']==0:
  netParams.connParams['EV1->IV1']['connList'] = blistEV1toIV1
  sim.topologicalConns['EV1->IV1'] = {}
  sim.topologicalConns['EV1->IV1']['blist'] = blistEV1toIV1
  sim.topologicalConns['EV1->IV1']['coords'] = connCoordsEV1toIV1
else:
  netParams.connParams['EV1->IV1']['convergence'] = prob2conv(0.0225, dnumc['EV1'])

if 'EPreADirPops' in dconf['net'] and 'IPreADirPops' in dconf['net']:
  if 'ID' in dconf['net']['allpops']:
    if dnumc['ID']>0:
      EDirPops = dconf['net']['EPreADirPops']
      IDirPops = dconf['net']['IPreADirPops']
      for prety in EDirPops:
        for poty in IDirPops:
          netParams.connParams[prety+'->'+poty] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': prob2conv(0.0225, dnumc[prety]),
            'weight': 0.02 * cfg.EIGain,
            'delay': 2,
            'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}

netParams.connParams['EV4->IV4'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'IV4'},
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}

if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['EV4->IV4']['connList'] = blistEV4toIV4
  sim.topologicalConns['EV4->IV4'] = {}
  sim.topologicalConns['EV4->IV4']['blist'] = blistEV4toIV4
  sim.topologicalConns['EV4->IV4']['coords'] = connCoordsEV4toIV4
else: 
  netParams.connParams['EV4->IV4']['convergence'] = prob2conv(0.0225, dnumc['EV4'])

netParams.connParams['EMT->IMT'] = {
        'preConds': {'pop': 'EMT'},
        'postConds': {'pop': 'IMT'},
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}

if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['EMT->IMT']['connList'] = blistEMTtoIMT
  sim.topologicalConns['EMT->IMT'] = {}
  sim.topologicalConns['EMT->IMT']['blist'] = blistEMTtoIMT
  sim.topologicalConns['EMT->IMT']['coords'] = connCoordsEMTtoIMT
else: 
  netParams.connParams['EMT->IMT']['convergence'] = prob2conv(0.0225, dnumc['EMT'])

"""  
for prety in EPreAPops:
  for epoty in EMotorPops:
    poty = 'IM' + epoty[2:]
    k = prety+'->'+poty
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': prob2conv(0.01, dnumc[prety]),
      'weight': 0.001 * cfg.EIGain,
      'delay': 2,
      'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
    if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
    elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}    
"""

for prety in ['EA']:
  k = prety+'->IA'
  netParams.connParams[k] = {
    'preConds': {'pop': prety},
    'postConds': {'pop': 'IA'},
    'convergence': prob2conv(0.125/2, dnumc[prety]),
    'weight': 0.02 * cfg.EIGain,
    'delay': 2,
    'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
  if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
    netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
    netParams.connParams[k]['weight'] = getInitWeight(0.02 * cfg.EIGain)
  elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
    netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
    netParams.connParams[k]['weight'] = getInitWeight(0.02 * cfg.EIGain)    


for prety in EMotorPops:
  k = prety+'->IM'
  netParams.connParams[k] = {
    'preConds': {'pop': prety},
    'postConds': {'pop': 'IM'},
    'convergence': prob2conv(0.125/2, dnumc[prety]),
    'weight': 0.02 * cfg.EIGain,
    'delay': 2,
    'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
  if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
    netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
    netParams.connParams[k]['weight'] = getInitWeight(0.02 * cfg.EIGain)
  elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
    netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
    netParams.connParams[k]['weight'] = getInitWeight(0.02 * cfg.EIGain)    

# reciprocal inhibition - only active when all relevant populations created
for prety in EMotorPops:
  for epoty in EMotorPops:
    if epoty == prety: continue # no self inhib here
    poty = 'IM' + epoty[2:] # change name to interneuron
    k = prety + '->' + poty
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': prob2conv(dconf['net']['EMIRecipProb'], dnumc[prety]),
      'weight': dconf['net']['EMIRecipWght'] * cfg.EIGain,
      'delay': 2,
      'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
    if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
      netParams.connParams[k]['weight'] = getInitWeight(dconf['net']['EMIRecipWght'] * cfg.EIGain)      
    elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
      netParams.connParams[k]['weight'] = getInitWeight(dconf['net']['EMIRecipWght'] * cfg.EIGain)            

    
#Local inhibition
#I to E within area
if dnumc['ER']>0:
  netParams.connParams['IR->ER'] = {
          'preConds': {'pop': 'IR'},
          'postConds': {'pop': 'ER'},
          'weight': 0.2 * cfg.IEGain,
          'delay': 2,
          'synMech': 'GABA', 'sec':'soma', 'loc':0.5}
  
  if VTopoI and dconf['sim']['useReducedNetwork']==0: 
    netParams.connParams['IR->ER']['connList'] = blistIRtoER
    sim.topologicalConns['IR->ER'] = {}
    sim.topologicalConns['IR->ER']['blist'] = blistIRtoER
    sim.topologicalConns['IR->ER']['coords'] = connCoordsIRtoER
  else: 
    netParams.connParams['IR->ER']['convergence'] = prob2conv(0.0625, dnumc['ER'])
  
netParams.connParams['IV1->EV1'] = {
  'preConds': {'pop': 'IV1'},
  'postConds': {'pop': 'EV1'},
  'weight': 0.2 * cfg.IEGain,
  'delay': 2,
  'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['IV1->EV1']['connList'] = blistIV1toEV1
  sim.topologicalConns['IV1->EV1'] = {}
  sim.topologicalConns['IV1->EV1']['blist'] = blistIV1toEV1
  sim.topologicalConns['IV1->EV1']['coords'] = connCoordsIV1toEV1
else: 
  netParams.connParams['IV1->EV1']['convergence'] = prob2conv(0.25, dnumc['IV1'])  

if 'EPreADirPops' in dconf['net'] and 'IPreADirPops' in dconf['net']:
  if 'ID' in dconf['net']['allpops']:
    if dnumc['ID']>0:
      EDirPops = dconf['net']['EPreADirPops']
      IDirPops = dconf['net']['IPreADirPops']
      for prety in IDirPops:
        for poty in EDirPops:
          netParams.connParams[prety+'->'+poty] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': prob2conv(0.25, dnumc['ID']),
            'weight': 0.2 * cfg.IEGain,
            'delay': 2,
            'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

netParams.connParams['IV4->EV4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'EV4'},
        'weight': 0.2 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['IV4->EV4']['connList'] = blistIV4toEV4
  sim.topologicalConns['IV4->EV4'] = {}
  sim.topologicalConns['IV4->EV4']['blist'] = blistIV4toEV4
  sim.topologicalConns['IV4->EV4']['coords'] = connCoordsIV4toEV4
else: 
  netParams.connParams['IV4->EV4']['convergence'] = prob2conv(0.25, dnumc['IV4'])

netParams.connParams['IMT->EMT'] = {
        'preConds': {'pop': 'IMT'},
        'postConds': {'pop': 'EMT'},
        #'connList': blistIMTtoEMT,
        'weight': 0.2 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['IMT->EMT']['connList'] = blistIMTtoEMT
  sim.topologicalConns['IMT->EMT'] = {}
  sim.topologicalConns['IMT->EMT']['blist'] = blistIMTtoEMT
  sim.topologicalConns['IMT->EMT']['coords'] = connCoordsIMTtoEMT
else: 
  netParams.connParams['IMT->EMT']['convergence'] = prob2conv(0.25, dnumc['IMT'])

for poty in EMotorPops: # I -> E for motor populations
  netParams.connParams['IM->'+poty] = {
    'preConds': {'pop': 'IM'},
    'postConds': {'pop': poty},
    'convergence': prob2conv(0.125, dnumc['IM']),
    'weight': 0.2 * cfg.IEGain,
    'delay': 2,
    'synMech': 'GABA', 'sec':'soma', 'loc':0.5}
  prety = 'IM' + poty[2:] # for reciprocal inhibition between EM populations
  netParams.connParams[prety+'->'+poty] = {
    'preConds': {'pop': prety},
    'postConds': {'pop': poty},
    'convergence': prob2conv(0.125, dnumc[prety]),
    'weight': 0.2 * cfg.IEGain,
    'delay': 2,
    'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

for poty in ['EA']:
  netParams.connParams['IA->'+poty] = {
    'preConds': {'pop': 'IA'},
    'postConds': {'pop': poty},
    'convergence': prob2conv(0.125, dnumc['IA']),
    'weight': 0.2 * cfg.IEGain,
    'delay': 2,
    'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

#I to I
for IType in ITypes:
  if IType not in dnumc: continue
  if dnumc[IType] <= 0: continue
  netParams.connParams[IType+'->'+IType] = {
    'preConds': {'pop': IType},
    'postConds': {'pop': IType},
    'convergence': prob2conv(0.25, dnumc[IType]),
    'weight': 0.05 * cfg.IIGain, 
    'delay': 2,
    'synMech': 'GABA', 'sec':'soma', 'loc':0.5}  

#E to E feedforward connections - AMPA,NMDA

lprety,lpoty,lblist,lconnsCoords,lprob = [],[],[],[],[]
if dconf['sim']['useReducedNetwork']==0:
  if dnumc['ER']>0:
    lprety.append('ER')
    lpoty.append('EV1')
    lblist.append(blistERtoEV1)
    lprob.append(dconf['net']['EREV1Prob'])
    lconnsCoords.append(connCoordsERtoEV1)
  lprety.append('EV1'); lpoty.append('EV4'); lblist.append(blistEV1toEV4); lprob.append(dconf['net']['EV1EV4Prob']); lconnsCoords.append(connCoordsEV1toEV4)
  lprety.append('EV4'); lpoty.append('EMT'); lblist.append(blistEV4toEMT); lprob.append(dconf['net']['EV4EMTProb']); lconnsCoords.append(connCoordsEV4toEMT)
  for prety,poty,blist,prob,connCoords in zip(lprety,lpoty,lblist,lprob,lconnsCoords):  
    for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],[dconf['net']['EEVisWghtAM']*cfg.EEGain, dconf['net']['EEVisWghtNM']*cfg.EEGain]):
      k = strty+prety+'->'+strty+poty
      if poty == 'EMT': weight *= 3
      netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'weight': weight ,
            'delay': 2,
            'synMech': synmech,'sec':EExcitSec, 'loc':0.5}
      if VTopoI: # topological connections
        netParams.connParams[k]['connList'] = blist
        sim.topologicalConns[prety+'->'+poty] = {}
        sim.topologicalConns[prety+'->'+poty]['blist'] = blist
        sim.topologicalConns[prety+'->'+poty]['coords'] = connCoords
      else: # random connectivity
        netParams.connParams[k]['convergence'] = prob2conv(prob,dnumc[prety])
      if dconf['net']['RLconns']['Visual'] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['weight'] = getInitWeight(weight) # make sure non-uniform weights
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif dconf['net']['STDPconns']['Visual'] and dSTDPparams[synmech]['STDPon']:
        netParams.connParams[k]['weight'] = getInitWeight(weight) # make sure non-uniform weights
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

"""
# these all have 0 weight, dont set them up - though no harm in setting them up
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

if dconf['net']['VisualFeedback'] and dconf['sim']['useReducedNetwork']==0:
  # visual area feedback connections
  pretyList = ['EV1','EV4','EMT']
  potyList = ['ER','EV1','EV4']
  allconnList = [blistEV1toER,blistEV4toEV1,blistEMTtoEV4]
  allconnCoords = [connCoordsEV1toER,connCoordsEV4toEV1,connCoordsEMTtoEV4]
  for prety,poty,connList,connCoords in zip(pretyList,potyList,allconnList,allconnCoords):
    if dnumc[prety] <= 0 or dnumc[poty] <= 0: continue # skip empty pops
    for strty,synmech,synweight in zip(['','n'],['AMPA', 'NMDA'],[dconf['net']['EEMWghtAM']*cfg.EEGain, dconf['net']['EEMWghtNM']*cfg.EEGain]):
        k = strty+prety+'->'+strty+poty
        netParams.connParams[k] = {
          'preConds': {'pop': prety},
          'postConds': {'pop': poty},
          'connList': connList,
          'weight': getInitWeight(synweight),
          'delay': 2,
          'synMech': synmech,'sec':EExcitSec, 'loc':0.5} # 'weight' should be fixed
        sim.topologicalConns[prety+'->'+poty] = {}
        sim.topologicalConns[prety+'->'+poty]['blist'] = connList
        sim.topologicalConns[prety+'->'+poty]['coords'] = connCoords  
        if dconf['net']['RLconns']['FeedbackLocNeurons'] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
        elif dconf['net']['STDPconns']['FeedbackLocNeurons'] and dSTDPparams[synmech]['STDPon']:
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}          
  #I to E feedback connections
  netParams.connParams['IV1->ER'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'ER'},
        'connList': blistIV1toER,
        'weight': 0.02 * cfg.IEGain, 
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}
  sim.topologicalConns['IV1->ER'] = {}
  sim.topologicalConns['IV1->ER']['blist'] = blistIV1toER
  sim.topologicalConns['IV1->ER']['coords'] = connCoordsIV1toER
  netParams.connParams['IV4->EV1'] = {
          'preConds': {'pop': 'IV4'},
          'postConds': {'pop': 'EV1'},
          'connList': blistIV4toEV1,
          'weight': 0.02 * cfg.IEGain, 
          'delay': 2,
          'synMech': 'GABA','sec':'soma', 'loc':0.5}
  sim.topologicalConns['IV4->EV1'] = {}
  sim.topologicalConns['IV4->EV1']['blist'] = blistIV4toEV1
  sim.topologicalConns['IV4->EV1']['coords'] = connCoordsIV4toEV1
  netParams.connParams['IMT->EV4'] = {
          'preConds': {'pop': 'IMT'},
          'postConds': {'pop': 'EV4'},
          'connList': blistIMTtoEV4,
          'weight': 0.02 * cfg.IEGain, 
          'delay': 2,
          'synMech': 'GABA','sec':'soma', 'loc':0.5}
  sim.topologicalConns['IMT->EV4'] = {}
  sim.topologicalConns['IMT->EV4']['blist'] = blistIMTtoEV4
  sim.topologicalConns['IMT->EV4']['coords'] = connCoordsIMTtoEV4

#I to I - between areas
if dconf['sim']['useReducedNetwork']==0:
  netParams.connParams['IV1->IV4'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistIV1toIV4,
        'weight': 0.0075 * cfg.IIGain,
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}
  sim.topologicalConns['IV1->IV4'] = {}
  sim.topologicalConns['IV1->IV4']['blist'] = blistIV1toIV4
  sim.topologicalConns['IV1->IV4']['coords'] = connCoordsIV1toIV4
  netParams.connParams['IV4->IMT'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IMT'},
        'connList': blistIV4toIMT,
        'weight': 0.0075 * cfg.IIGain,
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}
  sim.topologicalConns['IV4->IMT'] = {}
  sim.topologicalConns['IV4->IMT']['blist'] = blistIV4toIMT
  sim.topologicalConns['IV4->IMT']['coords'] = connCoordsIV4toIMT

if dconf['sim']['useReducedNetwork']==0:
  if dconf['architecturePreAtoA']['useTopological']:
    for prety in EPreAPops:
      if dnumc[prety] <= 0: continue
      for poty in ['EA']:
        if dnumc[poty] <= 0: continue
        try:
          div = dconf['net']['alltopoldivcons'][prety][poty]
        except:
          div = 3
        # BE CAREFUL. THERE IS ALWAYS A CHANCE TO USE dnumc[prety] nad dnumc[poty] that produces inaccuracies.
        # works fine if used in multiples (400->100; 100->400; 100->100).
        blist = []
        connCoords = []        
        if dconf['net']['allpops'][prety]==dconf['net']['allpops'][poty] or dconf['net']['allpops'][prety]>dconf['net']['allpops'][poty]: 
          blist, connCoords = connectLayerswithOverlap(NBpreN=dnumc[prety],NBpostN=dnumc[poty],overlap_xdir = dtopolconvcons[prety][poty], \
                                         padded_preneurons_xdir = dnumc_padx[prety], padded_postneurons_xdir = dnumc_padx[poty])
        elif dconf['net']['allpops'][prety]<dconf['net']['allpops'][poty]:
          blist, connCoords = connectLayerswithOverlapDiv(NBpreN=dnumc[prety],NBpostN=dnumc[poty],overlap_xdir = dtopoldivcons[prety][poty], \
                                            padded_preneurons_xdir = dnumc_padx[prety], padded_postneurons_xdir = dnumc_padx[poty])
          print(prety,poty,blist)
        sim.topologicalConns[prety+'->'+poty] = {}
        sim.topologicalConns[prety+'->'+poty]['blist'] = blist
        sim.topologicalConns[prety+'->'+poty]['coords'] = connCoords
        # print(prety,poty,len(blist),len(np.unique(np.array(blist)[:,0])),len(np.unique(np.array(blist)[:,1])))
        lsynw = [dconf['net']['EEAWghtAM']*cfg.EEGain, dconf['net']['EEAWghtNM']*cfg.EEGain]
        if prety in EVPops:
          lsynw = [dconf['net']['VISAWghtAM']*cfg.EEGain, dconf['net']['VISAWghtNM']*cfg.EEGain]      
        for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],lsynw):        
          k = strty+prety+'->'+strty+poty
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'connList': blist, 
            'weight': getInitWeight(weight),
            'delay': 2,
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5
          }
          useRL = useSTDP = False
          if prety in dconf['net']['EPreADirPops']:
            if dconf['net']['RLconns']['FeedForwardDirNtoA']: useRL = True
            if dconf['net']['STDPconns']['FeedForwardDirNtoA']: useSTDP = True          
          if prety in dconf['net']['EPreALocPops']:
            if dconf['net']['RLconns']['FeedForwardLocNtoA']: useRL = True
            if dconf['net']['STDPconns']['FeedForwardLocNtoA']: useSTDP = True
          if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif dSTDPparams[synmech]['STDPon'] and useSTDP:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}          
  elif dconf['architecturePreAtoA']['useProbabilistic']:
    # Add connections from lower and higher visual areas to motor cortex and direct connections between premotor to motor areas
    for prety in EPreAPops:
      if dnumc[prety] <= 0: continue
      EEAProb = dconf['net']['EEAProb'] # feedforward connectivity
      lsynw = [dconf['net']['EEAWghtAM']*cfg.EEGain, dconf['net']['EEAWghtNM']*cfg.EEGain]
      if prety in EVPops:
        lsynw = [dconf['net']['VISAWghtAM']*cfg.EEGain, dconf['net']['VISAWghtNM']*cfg.EEGain]    
      for poty in ['EA']:
        if dnumc[poty] <= 0: continue
        for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],lsynw):
          k = strty+prety+'->'+strty+poty
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': prob2conv(EEAProb, dnumc[prety]),
            'weight': getInitWeight(weight),
            'delay': 2,
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5
          }
          useRL = useSTDP = False
          if prety in dconf['net']['EPreADirPops']:
            if dconf['net']['RLconns']['FeedForwardDirNtoA']: useRL = True
            if dconf['net']['STDPconns']['FeedForwardDirNtoA']: useSTDP = True          
          if prety in dconf['net']['EPreALocPops']:
            if dconf['net']['RLconns']['FeedForwardLocNtoA']: useRL = True
            if dconf['net']['STDPconns']['FeedForwardLocNtoA']: useSTDP = True
          if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif dSTDPparams[synmech]['STDPon'] and useSTDP:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}                  
else:
  print(cLV1toEA)
  synmech = 'AMPA'
  weight = dconf['net']['EEAWghtAM']*cfg.EEGain
  netParams.connParams['EV1->EA'] = {
      'preConds': {'pop': 'EV1'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1toEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  if dconf['net']['RLconns']['FeedForwardLocNtoA'] and dSTDPparamsRL[synmech]['RLon']: 
    netParams.connParams['EV1->EA']['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
  elif dconf['net']['STDPconns']['FeedForwardLocNtoA'] and dSTDPparams[synmech]['STDPon']:
    netParams.connParams['EV1->EA']['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}
  netParams.connParams['EV1DE->EA'] = {
      'preConds': {'pop': 'EV1DE'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1DEtoEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  netParams.connParams['EV1DNE->EA'] = {
      'preConds': {'pop': 'EV1DNE'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1DNEtoEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  netParams.connParams['EV1DN->EA'] = {
      'preConds': {'pop': 'EV1DN'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1DNtoEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  netParams.connParams['EV1DNW->EA'] = {
      'preConds': {'pop': 'EV1DNW'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1DNWtoEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  netParams.connParams['EV1DW->EA'] = {
      'preConds': {'pop': 'EV1DW'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1DWtoEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  netParams.connParams['EV1DSW->EA'] = {
      'preConds': {'pop': 'EV1DSW'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1DSWtoEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  netParams.connParams['EV1DS->EA'] = {
      'preConds': {'pop': 'EV1DS'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1DStoEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  netParams.connParams['EV1DSE->EA'] = {
      'preConds': {'pop': 'EV1DSE'},
      'postConds': {'pop': 'EA'},
      'connList': cLV1DSEtoEA, 
      'weight': getInitWeight(weight),
      'delay': 2,
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5
  }
  ldirconns = ['EV1DE->EA','EV1DNE->EA','EV1DN->EA','EV1DNW->EA','EV1DW->EA','EV1DSW->EA','EV1DS->EA','EV1DSE->EA']
  if dconf['net']['RLconns']['FeedForwardDirNtoA'] and dSTDPparamsRL[synmech]['RLon']:
    for dirconns in ldirconns: 
      netParams.connParams[dirconns]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
  elif dconf['net']['STDPconns']['FeedForwardDirNtoA'] and dSTDPparams[synmech]['STDPon']:
    for dirconns in ldirconns:
      netParams.connParams[dirconns]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}


# Add connections from lower and higher visual areas to motor cortex and direct connections between premotor to motor areas
for prety in ['EA']:
  if dnumc[prety] <= 0: continue
  EEMProb = dconf['net']['EEMProb'] # feedforward connectivity
  lsynw = [dconf['net']['EEMWghtAM']*cfg.EEGain, dconf['net']['EEMWghtNM']*cfg.EEGain]
  for poty in EMotorPops:
    if dnumc[poty] <= 0: continue
    for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],lsynw):
      k = strty+prety+'->'+strty+poty
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': prob2conv(EEMProb, dnumc[prety]),
        'weight': getInitWeight(weight),
        'delay': 2,
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5
      }
      useRL = useSTDP = False
      if dconf['net']['RLconns']['FeedForwardAtoM']: useRL = True
      if dconf['net']['STDPconns']['FeedForwardAtoM']: useSTDP = True          
      if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif dSTDPparams[synmech]['STDPon'] and useSTDP:
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}                  
                    
# add recurrent plastic connectivity within EA populations
EEARecProb = dconf['net']['EEARecProb']
if EEARecProb > 0.0:
  for prety in ['EA']:
    for poty in ['EA']:
      for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],[dconf['net']['EEAWghtAM']*cfg.EEGain, dconf['net']['EEAWghtNM']*cfg.EEGain]):
        k = strty+prety+'->'+strty+poty
        netParams.connParams[k] = {
          'preConds': {'pop': prety},
          'postConds': {'pop': poty},
          'convergence': prob2conv(EEARecProb, dnumc[prety]),
          'weight': getInitWeight(weight),
          'delay': 2,
          'synMech': synmech,
          'sec':EExcitSec, 'loc':0.5
        }
        if dconf['net']['RLconns']['RecurrentANeurons'] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
        elif dconf['net']['STDPconns']['RecurrentANeurons'] and dSTDPparams[synmech]['STDPon']:
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}            

          
# add recurrent plastic connectivity within EM populations
EEMRecProb = dconf['net']['EEMRecProb']
if EEMRecProb > 0.0:
  for prety in EMotorPops:
    for poty in EMotorPops:
      if prety==poty or dconf['net']['EEMRecProbCross']: # same types or allowing cross EM population connectivity
        for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],[dconf['net']['EEMWghtAM']*cfg.EEGain, dconf['net']['EEMWghtNM']*cfg.EEGain]):
          k = strty+prety+'->'+strty+poty
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': prob2conv(EEMRecProb, dnumc[prety]),
            'weight': getInitWeight(weight),
            'delay': 2,
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5
          }
          if dconf['net']['RLconns']['RecurrentMNeurons'] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif dconf['net']['STDPconns']['RecurrentMNeurons'] and dSTDPparams[synmech]['STDPon']:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}            

# add feedback plastic connectivity from EM populations to association populations
EEMFeedbackProb = dconf['net']['EEMFeedbackProb']
if EEMFeedbackProb > 0.0:
  for prety in EMotorPops:
    for poty in ['EA']:
        for strty,synmech,weight in zip(['','n'],['AMPA', 'NMDA'],[dconf['net']['EEMFeedbackWghtAM']*cfg.EEGain, dconf['net']['EEMFeedbackWghtNM']*cfg.EEGain]):
          k = strty+prety+'->'+strty+poty
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': prob2conv(EEMFeedbackProb, dnumc[prety]),
            'weight': getInitWeight(weight),
            'delay': 2,
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5
          }
          useRL = useSTDP = False
          if poty in dconf['net']['EPreADirPops']:
            if dconf['net']['RLconns']['FeedbackMtoDirN']: useRL = True
            if dconf['net']['STDPconns']['FeedbackMtoDirN']: useSTDP = True
          if poty in dconf['net']['EPreALocPops']:
            if dconf['net']['RLconns']['FeedbackMtoLocN']: useRL = True
            if dconf['net']['STDPconns']['FeedbackMtoLocN']: useSTDP = True            
          if useRL and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif useSTDP and dSTDPparams[synmech]['STDPon']:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}            

fconn = 'data/'+dconf['sim']['name']+'synConns.pkl'
pickle.dump(sim.topologicalConns, open(fconn, 'wb'))            
###################################################################################################################################

sim.AIGame = None # placeholder

lsynweights = [] # list of syn weights, per node

dsumWInit = {}

def getSumAdjustableWeights (sim):
  dout = {}
  for cell in sim.net.cells:
    W = N = 0.0
    for conn in cell.conns:
      if 'hSTDP' in conn:
        W += float(conn['hObj'].weight[0])
        N += 1
    if N > 0:
      dout[cell.gid] = W / N
  # print('getSumAdjustableWeights len=',len(dout))
  return dout

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
        #hstdp = conn.get('hSTDP')
        lsynweights.append([t,conn.preGid,cell.gid,float(conn['hObj'].weight[0])])#,hstdp.cumreward])
  return len(lcell)
                    
def recordAdjustableWeights (sim, t, lpop):
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

def getAverageAdjustableWeights (sim, lpop = EMotorPops):
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

def normalizeAdjustableWeights (sim, t, lpop):
  # normalize the STDP/RL weights during the simulation - called in trainAgent
  if dconf['net']['CellWNorm']['On']:
    # print('normalizing CellWNorm at t=',t)
    global dsumWInit
    dsumWCurr = getSumAdjustableWeights(sim) # get current sum of adjustable weight values
    MinFctr, MaxFctr = dconf['net']['CellWNorm']['MinFctr'], dconf['net']['CellWNorm']['MaxFctr']
    for cell in sim.net.cells:
      if cell.gid in dsumWInit:
        currW = dsumWCurr[cell.gid]
        initW = dsumWInit[cell.gid]
        if currW > 0 and currW != initW:
          fctrA = currW / initW
          dochange = False
          if fctrA < MinFctr:
            fctrB = (MinFctr * initW) / currW # factor to restore weights to boundary
            dochange = True
          elif fctrA > MaxFctr:
            fctrB = (MaxFctr * initW) / currW # factor to restore weights to boundary
            dochange = True
          # print('initW:',initW,'currW:',currW,'fctr:',fctr)
          if dochange:
            for conn in cell.conns:
              if 'hSTDP' in conn:
                conn['hObj'].weight[0] *= fctrB
  else:
    davg = getAverageAdjustableWeights(sim, lpop)
    try:
      dfctr = {}
      # normalize weights across populations to avoid bias    
      initw = dconf['net']['EEMWghtAM'] # initial average weight <<-- THAT ASSUMES ONLY USING NORM ON EM POPULATIONS!
      if dconf['net']['EEMPopNorm']:
        curravgw = np.mean([davg[k] for k in lpop]) # current average weight
        if curravgw <= 0.0: curravgw = initw
        for k in lpop: dfctr[k] = initw / curravgw
      else:
        for k in lpop:
          if davg[k] < dconf['net']['EEMWghtThreshMin']:
            dfctr[k] = dconf['net']['EEMWghtThreshMin'] / davg[k]
          elif davg[k] > dconf['net']['EEMWghtThreshMax']:
            dfctr[k] = dconf['net']['EEMWghtThreshMax'] / davg[k]
          else:
            dfctr[k] = 1.0
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

def getSpikesWithInterval (trange = None, neuronal_pop = None):
  if len(neuronal_pop) < 1: return 0.0
  spkts = sim.simData['spkt']
  spkids = sim.simData['spkid']
  pop_spikes = 0
  if len(spkts)>0:
    for i in range(len(spkids)):
      if trange[0] <= spkts[i] <= trange[1] and spkids[i] in neuronal_pop:
        pop_spikes += 1
  return pop_spikes

NBsteps = 0 # this is a counter for recording the plastic weights
epCount = []
proposed_actions = [] 
total_hits = [] #numbertimes ball is hit by racket as ball changes its direction and player doesn't lose a score (assign 1). if player loses
dSTDPmech = {} # dictionary of list of STDP mechanisms

def InitializeInputRates ():
  # update the source firing rates for the ER neuron population, based on image contents
  # also update the firing rates for the direction sensitive neurons based on image contents
  if dnumc['ER']>0:
    lratepop = ['ER', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']
  else:
    lratepop = ['EV1', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']  
  for pop in lratepop:
    lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells
    for cell in lCell:  
      for stim in cell.stims:
        if stim['source'] == 'stimMod':
          stim['hObj'].interval = 1e12

def updateInputRates ():
  # update the source firing rates for the ER neuron population, based on image contents
  # also update the firing rates for the direction sensitive neurons based on image contents
  if dnumc['ER']>0:
    lratepop = ['ER', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']
  else:
    lratepop = ['EV1', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']
  # sim.pc.barrier()

  #root = 0 # any specific rank
  #src = rank if rank == root else None
  #dest = pc.py_broadcast(src, root)

  #src = dict(sim.AIGame.dFiringRates) if sim.rank == 0 else None
  #print('rank=',sim.rank,'src=',src)


  # this py_alltoall seems to work, but is apparently not as fast as py_broadcast (which has problems - wrong-sized array sometimes!)
  root = 0
  nhost = sim.pc.nhost()
  src = [sim.AIGame.dFiringRates]*nhost if sim.rank == root else [None]*nhost
  dFiringRates = sim.pc.py_alltoall(src)[0]
  if sim.rank == 0: dFiringRates = sim.AIGame.dFiringRates
  # print(sim.rank,'dFiringRates:',dFiringRates)

  """ # for some reason this code does not work - returns wrong sized array on dest sometimes, saw it produce MPI error too
  if sim.rank == 0:
    dFiringRates = sim.AIGame.dFiringRates
    sim.pc.py_broadcast(dFiringRates, 0)
  else:
    dFiringRates = sim.pc.py_broadcast(None, 0)
    #dFiringRates = {pop:np.zeros(dnumc[pop]) for pop in allpops}
  # pop='EV1'; print('rank=',sim.rank,pop,len(dFiringRates[pop]))
  """
  
  """ # this also sometimes produces wrong-sized array on dest
  sim.pc.barrier()
  if sim.rank == 0:
    #print('rank0:',len(sim.AIGame.dFiringRates['EV1']))
    #dFiringRates = sim.pc.py_broadcast(dict(sim.AIGame.dFiringRates), 0)
    dFiringRates = sim.AIGame.dFiringRates
    for k in sim.AIGame.lratepop:
      sim.pc.broadcast(sim.AIGame.dFVec[k].from_python(dFiringRates[k]),0)
      if dconf['verbose'] > 1: print('Firing Rates of',k,np.where(dFiringRates[k]==np.amax(dFiringRates[k])),np.amax(dFiringRates[k]))
      print('rank=',sim.rank,k,len(dFiringRates[k]))
  else:
    #dFiringRates = sim.pc.py_broadcast(None, 0) # OrderedDict()
    dFiringRates = OrderedDict()
    for pop in lratepop:
      vec = h.Vector()
      destSz = sim.pc.broadcast(vec,0)
      print('rank=',sim.rank,'destSz=',destSz)
      dFiringRates[pop] = vec.to_python()
      if dconf['verbose'] > 1:
        print(sim.rank,'received firing rates:',np.where(dFiringRates[pop]==np.amax(dFiringRates[pop])),np.amax(dFiringRates[pop]))
      print('rank=',sim.rank,pop,len(dFiringRates[pop]))
  sim.pc.barrier()
  """
  
  # update input firing rates for stimuli to ER,EV1 and direction sensitive cells
  for pop in lratepop:
    lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells
    offset = sim.simData['dminID'][pop]
    #if dconf['verbose'] > 1: print(sim.rank,'updating len(',pop,len(lCell),'source firing rates. len(dFiringRates)=',len(dFiringRates[pop]))
    for cell in lCell:  
      for stim in cell.stims:
        if stim['source'] == 'stimMod':
          #rind = random.randint(0,2)
          #fchoices = [20,50,1000] #20 will represent 50Hz and so on
          # print(pop,len(dFiringRates[pop]),int(cell.gid-offset),cell.gid,offset)
          if dFiringRates[pop][int(cell.gid-offset)]==0:
            stim['hObj'].interval = 1e12
          else:
            stim['hObj'].interval = 1000.0/dFiringRates[pop][int(cell.gid-offset)] #40 #fchoices[rind] #10 #
          #sim.AIGame.dAllFiringRates[-1][pop][int(cell.gid-offset)] = fchoices[rind] 
          #print('cell GID: ', int(cell.gid), 'vs cell ID with offset: ', int(cell.gid-R_offset)) # interval in ms as a function of rate; is cell.gid correct index??? 

# homeostasis based on Sanda et al. 2017.

def initTargetFR (sim,dMINinitTargetFR,dMAXinitTargetFR):
  sim.dMINTargetFR = {}
  sim.dMAXTargetFR = {}
  sim.dHPlastPops = list(dMINinitTargetFR.keys())
  for pop in dMINinitTargetFR.keys():
    for gid in sim.net.pops[pop].cellGids:
      sim.dMINTargetFR[gid] = dMINinitTargetFR[pop]
      sim.dMAXTargetFR[gid] = dMAXinitTargetFR[pop]

def initTargetW (sim,lpop,synType='AMPA'):
  sim.dTargetW = {}
  for pop in lpop:
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells in a pop
    for cell in lcell:
      cCellW = 0
      for conn in cell.conns:
        if conn.synMech==synType and 'hSTDP' in conn:
          cCellW+=conn['hObj'].weight[0]
      sim.dTargetW[cell.gid] = cCellW

def getFiringRateWithInterval (trange = None, neuronal_pop = None):
  if len(neuronal_pop) < 1: return 0.0
  spkts = np.array(sim.simData['spkt'])
  spkids = np.array(sim.simData['spkid'])
  ctspkids = spkids[(spkts>=trange[0])&(spkts <= trange[1])]
  pop_firingrates = {}
  if len(spkts)>0:
    for cellid in neuronal_pop:
      pop_firingrates[cellid] = 1000.0*len(np.where(ctspkids==cellid))/(trange[1]-trange[0])
  return pop_firingrates

def getFiringRateWithIntervalAllNeurons (sim, trange, allpops):
  lgid = []
  for pop in allpops:
    for gid in sim.net.pops[pop].cellGids:
      lgid.append(gid)
  sim.dFR = getFiringRateWithInterval(trange = trange, neuronal_pop = lgid)

def adjustTargetWBasedOnFiringRates (sim):
  dshift = dconf['net']['homPlast']['dshift'] # shift in weights to push within min,max firing rate bounds
  for gid,cTargetW in sim.dTargetW.items():
    cTargetFRMin, cTargetFRMax = sim.dMINTargetFR[gid], sim.dMAXTargetFR[gid]
    if gid not in sim.dFR: continue
    cFR = sim.dFR[gid] # current cell firing rate
    if cFR>cTargetFRMax: # above max threshold firing rate
      if dshift == 0.0:
        sim.dTargetW[gid] *= cTargetFRMax / cFR
      else:
        sim.dTargetW[gid] -= dshift
    elif cFR<cTargetFRMin: # below min threshold firing rate
      if dshift == 0:
        sim.dTargetW[gid] *= cTargetFRMin / cFR 
      else:
        sim.dTargetW[gid] += dshift
  return sim.dTargetW

def adjustWeightsBasedOnFiringRates (sim,lpop,synType='AMPA'):
  # normalize the STDP/RL weights during the simulation - called in trainAgent
  countScaleUps = 0
  countScaleDowns = 0
  for pop in lpop:
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells in a pop
    for cell in lcell:
      targetW = sim.dTargetW[cell.gid]
      cCellW = 0
      for conn in cell.conns:
        if conn.synMech==synType and 'hSTDP' in conn: # to make sure that only specific type of synapses are counted towards the sum.
          cCellW+=conn['hObj'].weight[0]
      if cCellW>0: # if no weight associated with the specific type of synapses, no need to asdjust weights.
        sfctr = targetW/cCellW
        if sfctr>1: countScaleUps += 1
        elif sfctr<1: countScaleDowns += 1
        for conn in cell.conns:
          if conn.synMech==synType and 'hSTDP' in conn:
            conn['hObj'].weight[0] *= sfctr
  print('CountScaleUps, CountScaleDowns: ', countScaleUps, countScaleDowns)

def trainAgent (t):
  """ training interface between simulation and game environment
  """
  global NBsteps, epCount, proposed_actions, total_hits, fid4, tstepPerAction
  vec = h.Vector()
  noWinner = False # no clear winner for the population firing rates (used below)
  if t<(tstepPerAction*dconf['actionsPerPlay']): # for the first time interval use randomly selected actions
    actions =[]
    for _ in range(int(dconf['actionsPerPlay'])):
      action = dconf['movecodes'][random.randint(0,len(dconf['movecodes'])-1)]
      actions.append(action)
  else: #the actions should be based on the activity of motor cortex (EMUP, EMDOWN)
    F_UPs = []
    F_DOWNs = []
    F_STAYs = []
    for ts in range(int(dconf['actionsPerPlay'])):
      ts_beg = t-tstepPerAction*(dconf['actionsPerPlay']-ts-1) 
      ts_end = t-tstepPerAction*(dconf['actionsPerPlay']-ts)
      F_UPs.append(getSpikesWithInterval([ts_end,ts_beg], sim.net.pops['EMUP'].cellGids))
      F_DOWNs.append(getSpikesWithInterval([ts_end,ts_beg], sim.net.pops['EMDOWN'].cellGids))
      if dnumc['EMSTAY']>0:
        F_STAYs.append(getSpikesWithInterval([ts_end,ts_beg], sim.net.pops['EMSTAY'].cellGids))
    sim.pc.allreduce(vec.from_python(F_UPs),1) #sum
    F_UPs = vec.to_python()
    sim.pc.allreduce(vec.from_python(F_DOWNs),1) #sum
    F_DOWNs = vec.to_python()
    if dnumc['EMSTAY']>0:
      sim.pc.allreduce(vec.from_python(F_STAYs),1) #sum
      F_STAYs = vec.to_python()
    if sim.rank==0:
      if fid4 is None: fid4 = open(sim.MotorOutputsfilename,'w')
      if dnumc['EMSTAY']>0:
        print('t=',round(t,2),' U,D,S spikes:', F_UPs, F_DOWNs, F_STAYs)
      else:
        print('t=',round(t,2),' U,D spikes:', F_UPs, F_DOWNs)
      fid4.write('%0.1f' % t)
      for ts in range(int(dconf['actionsPerPlay'])): fid4.write('\t%0.1f' % F_UPs[ts])
      for ts in range(int(dconf['actionsPerPlay'])): fid4.write('\t%0.1f' % F_DOWNs[ts])
      if dnumc['EMSTAY']>0:
        for ts in range(int(dconf['actionsPerPlay'])): fid4.write('\t%0.1f' % F_STAYs[ts])
      fid4.write('\n')
      actions = []
      randmove = 0
      if 'randmove' in dconf: randmove=dconf['randmove']
      if randmove:
        lmoves = list(dconf['moves'].values())
        for ts in range(int(dconf['actionsPerPlay'])): actions.append(lmoves[np.random.randint(0,len(lmoves))])
      else:
        for ts in range(int(dconf['actionsPerPlay'])):
          if dnumc['EMSTAY']>0: 
            if (F_UPs[ts]>F_DOWNs[ts]) and (F_UPs[ts]>F_STAYs[ts]): # UP WINS vs ALL
              actions.append(dconf['moves']['UP'])
            elif (F_DOWNs[ts]>F_UPs[ts]) and (F_DOWNs[ts]>F_STAYs[ts]): # DOWN WINS vs ALL
              actions.append(dconf['moves']['DOWN'])
            elif (F_STAYs[ts]>F_UPs[ts]) and (F_STAYs[ts]>F_DOWNs[ts]): # STAY WINS vs ALL
              actions.append(dconf['moves']['NOMOVE']) # No move
            else: 
              actions.append(dconf['moves']['NOMOVE']) # No move but also no clear winner
              noWinner = True
          else:
            if F_UPs[ts]>F_DOWNs[ts]: # UP WINS
              actions.append(dconf['moves']['UP'])
            elif F_DOWNs[ts]>F_UPs[ts]: # DOWN WINS
              actions.append(dconf['moves']['DOWN'])
            else:
              actions.append(dconf['moves']['NOMOVE']) # No move
              noWinner = True
  if sim.rank == 0:
    rewards, epCount, proposed_actions, total_hits = sim.AIGame.playGame(actions, epCount, t)
    print('t=',round(t,2),'proposed actions:', proposed_actions,', model actions:', actions)
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
      #rewards for hitting the ball
      critic_for_avoidingloss = 0
      if sum(total_hits)>0:
        critic_for_avoidingloss = dconf['rewardcodes']['hitBall'] #should be able to change this number from config file
      #rewards for following or avoiding the ball
      critic_for_following_ball = 0
      for caction, cproposed_action in zip(actions, proposed_actions):
        if cproposed_action == -1: # invalid action since e.g. ball not visible
          continue
        elif caction - cproposed_action == 0: # model followed proposed action - gets a reward
          critic_for_following_ball += dconf['rewardcodes']['followTarget'] #follow the ball
        else: # model did not follow proposed action - gets a punishment
          critic_for_following_ball += dconf['rewardcodes']['avoidTarget'] # didn't follow the ball
      #total rewards
      critic = critic + critic_for_avoidingloss + critic_for_following_ball
      rewards = [critic for i in range(len(rewards))]  # reset rewards to modified critic signal - should use more granular recording
    # use py_broadcast to avoid converting to/from Vector
    sim.pc.py_broadcast(critic, 0) # broadcast critic value to other nodes
    UPactions = np.sum(np.where(np.array(actions)==dconf['moves']['UP'],1,0))
    DOWNactions = np.sum(np.where(np.array(actions)==dconf['moves']['DOWN'],1,0))
    sim.pc.py_broadcast(UPactions,0) # broadcast UPactions
    sim.pc.py_broadcast(DOWNactions,0) # broadcast DOWNactions
    if dnumc['EMSTAY']>0:
      STAYactions = np.sum(np.where(np.array(actions)==dconf['moves']['NOMOVE'],1,0))
      sim.pc.py_broadcast(STAYactions,0) # broadcast STAYactions
    if dconf['sim']['anticipatedRL']:
      print(proposed_actions)
      sim.pc.py_broadcast(proposed_actions,0) # used proposed actions to target/potentiate the pop representing anticipated action.      
  else: # other workers
    critic = sim.pc.py_broadcast(None, 0) # receive critic value from master node
    UPactions = sim.pc.py_broadcast(None, 0)
    DOWNactions = sim.pc.py_broadcast(None, 0)
    if 'EMSTAY' in dconf['net']: STAYactions = sim.pc.broadcast(None, 0)
    if dconf['sim']['anticipatedRL']: proposed_actions = sim.pc.py_broadcast(None, 0)      
    if dconf['verbose']:
      if dnumc['EMSTAY']>0: print('UPactions: ', UPactions,'DOWNactions: ', DOWNactions, 'STAYactions: ', STAYactions)
      else: print('UPactions: ', UPactions,'DOWNactions: ', DOWNactions)
      
  if dconf['sim']['anticipatedRL']:
    cpaction = proposed_actions 
    anticipated_reward = dconf['rewardcodes']['followTarget']
    if cpaction==dconf['moves']['UP']:
      if dconf['verbose']: print('APPLY RL to EMUP')
      for STDPmech in dSTDPmech['EMUP']: STDPmech.reward_punish(float(anticipated_reward))
    elif cpaction==dconf['moves']['DOWN']:
      if dconf['verbose']: print('APPLY RL to EMDOWN')
      for STDPmech in dSTDPmech['EMDOWN']: STDPmech.reward_punish(float(anticipated_reward))
    elif cpaction==dconf['moves']['NOMOVE'] and dnumc['EMSTAY']>0:
      if dconf['verbose']: print('APPLY RL to EMSTAY')
      for STDPmech in dSTDPmech['EMSTAY']: STDPmech.reward_punish(float(anticipated_reward))
    else:
      print('No anticipated action for the input!!!')
  else:
    if critic != 0: # if critic signal indicates punishment (-1) or reward (+1)
      if sim.rank==0: print('t=',round(t,2),'- adjusting weights based on RL critic value:', critic)
      if dnumc['EMSTAY']>0:
        if dconf['sim']['targettedRL']:
          if not noWinner: # if there's a clear winner in terms of firing rates
            if UPactions>DOWNactions and UPactions>STAYactions: # UP WINS vs ALL
              if dconf['verbose']: print('APPLY RL to EMUP')
              for STDPmech in dSTDPmech['EMUP']: STDPmech.reward_punish(float(critic))
            elif DOWNactions>UPactions and DOWNactions>STAYactions: # DOWN WINS vs ALL
              if dconf['verbose']: print('APPLY RL to EMDOWN')
              for STDPmech in dSTDPmech['EMDOWN']: STDPmech.reward_punish(float(critic))
            elif STAYactions>UPactions and STAYactions>DOWNactions: # STAY WINS vs ALL
              if dconf['verbose']: print('APPLY RL to EMSTAY')
              for STDPmech in dSTDPmech['EMSTAY']: STDPmech.reward_punish(float(critic))
            elif UPactions==DOWNactions and UPactions>STAYactions: # UP and DOWN TIED
              if dconf['verbose']: print('APPLY RL to EMUP and EMDOWN')
              for STDPmech in dSTDPmech['EMUP']: STDPmech.reward_punish(float(critic))
              for STDPmech in dSTDPmech['EMDOWN']: STDPmech.reward_punish(float(critic))
            elif UPactions==STAYactions and UPactions>DOWNactions: # UP and STAY TIED
              if dconf['verbose']: print('APPLY RL to EMUP and EMSTAY')
              for STDPmech in dSTDPmech['EMUP']: STDPmech.reward_punish(float(critic))
              for STDPmech in dSTDPmech['EMSTAY']: STDPmech.reward_punish(float(critic))
            elif DOWNactions==STAYactions and DOWNactions>UPactions: # DOWN and STAY TIED
              if dconf['verbose']: print('APPLY RL to EMDOWN and EMSTAY')
              for STDPmech in dSTDPmech['EMDOWN']: STDPmech.reward_punish(float(critic))
              for STDPmech in dSTDPmech['EMSTAY']: STDPmech.reward_punish(float(critic))
            elif DOWNactions==STAYactions and UPactions==STAYactions: # ALL actions TIED
              if dconf['verbose']: print('APPLY RL to EMUP, EMDOWN and EMSTAY')
              for STDPmech in dSTDPmech['all']: STDPmech.reward_punish(float(critic))      
        else:
          if dconf['verbose']: print('APPLY RL to EMUP, EMDOWN and EMSTAY')
          for STDPmech in dSTDPmech['all']: STDPmech.reward_punish(float(critic))
      else:
        if dconf['sim']['targettedRL']:
          if not noWinner: # if there's a clear winner in terms of firing rates
            if UPactions==DOWNactions:
              if dconf['verbose']: print('APPLY RL to both EMUP and EMDOWN')
              for STDPmech in dSTDPmech['all']: STDPmech.reward_punish(float(critic))          
            elif UPactions>DOWNactions: # UP WINS vs DOWN
              if dconf['verbose']: print('APPLY RL to EMUP')
              for STDPmech in dSTDPmech['EMUP']: STDPmech.reward_punish(float(critic))
            elif DOWNactions>UPactions: # DOWN WINS vs UP
              if dconf['verbose']: print('APPLY RL to EMDOWN')
              for STDPmech in dSTDPmech['EMDOWN']: STDPmech.reward_punish(float(critic))        
        else:
          if dconf['verbose']: print('APPLY RL to both EMUP and EMDOWN')
          for STDPmech in dSTDPmech['all']: STDPmech.reward_punish(critic)
  if sim.rank==0:
    print('t=',round(t,2),' game rewards:', rewards) # only rank 0 has access to rewards      
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
  updateInputRates() # update firing rate of inputs to R population (based on image content)                
  NBsteps += 1
  if NBsteps % recordWeightStepSize == 0:
    if dconf['verbose'] > 0 and sim.rank==0:
      print('Weights Recording Time:', t, 'NBsteps:',NBsteps,'recordWeightStepSize:',recordWeightStepSize)
    recordAdjustableWeights(sim, t, lrecpop) 
    #recordWeights(sim, t)
  if NBsteps % normalizeWeightStepSize == 0:
    if dconf['verbose'] > 0 and sim.rank==0:
      print('Weight Normalize Time:', t, 'NBsteps:',NBsteps,'normalizeWeightStepSize:',normalizeWeightStepSize)
    normalizeAdjustableWeights(sim, t, lrecpop)
  if dconf['net']['homPlast']['On']:
    if NBsteps % dconf['net']['homPlast']['hsIntervalSteps'] == 0:
      hsInterval = tstepPerAction*dconf['actionsPerPlay']*dconf['net']['homPlast']['hsIntervalSteps']
      getFiringRateWithIntervalAllNeurons(sim, [t,t-hsInterval], sim.dHPlastPops) # call this function at hsInterval
      adjustTargetWBasedOnFiringRates(sim) # call this function at hsInterval
    if NBsteps % dconf['net']['homPlast']['updateIntervalSteps'] == 0:
      adjustWeightsBasedOnFiringRates(sim,sim.dHPlastPops,synType=dconf['net']['homPlast']['synType'])

def getAllSTDPObjects (sim):
  # get all the STDP objects from the simulation's cells
  Mpops = ['EMUP', 'EMDOWN']  
  if dnumc['EMSTAY']>0: Mpops.append('EMSTAY')
  dSTDPmech = {'all':[]} # dictionary of STDP objects keyed by type (all, for EMUP, EMDOWN populations)
  for pop in Mpops: dSTDPmech[pop] = []
  if dconf['sim']['targettedRL']: dcell = {pop:[] for pop in Mpops} # SN: exptl  
  for cell in sim.net.cells:
    for conn in cell.conns:
      STDPmech = conn.get('hSTDP')  # check if the connection has a NEURON STDP mechanism object
      if STDPmech:
        dSTDPmech['all'].append(STDPmech)
        for pop in Mpops:
          if cell.gid in sim.net.pops[pop].cellGids:
            dSTDPmech[pop].append(STDPmech)
            if dconf['sim']['targettedRL']: dcell[pop].append(conn.preGid) # SN: exptl presynaptic ID
  # SN: exptl
  #if 'hSTDP' not in conn: continue
  #cpreID = conn.preGid  #find preID
  #if type(cpreID) != int: continue
  if dconf['sim']['targettedRL'] > 1:
    for pop in Mpops: dcell[pop] = np.unique(dcell[pop])
    nhost = sim.pc.nhost()
    src = [dcell]*nhost
    dcellgid = sim.pc.py_alltoall(src)
    for dcell in dcellgid:
      for pop in Mpops:
        for cell in sim.net.cells:
          if cell.gid in dcell[pop]:
            for conn in cell.conns:
              STDPmech = conn.get('hSTDP')
              if STDPmech:
                if STDPmech not in dSTDPmech[pop]:
                  dSTDPmech[pop].append(STDPmech)
  # SN: exptl
  return dSTDPmech
        
# Alternate to create network and run simulation
# create network object and set cfg and net params; pass simulation config and network params as arguments
sim.initialize(simConfig = simConfig, netParams = netParams)

if sim.rank == 0:  # sim rank 0 specific init and backup of config file
  from aigame import AIGame
  sim.AIGame = AIGame() # only create AIGame on node 0
  # node 0 saves the json config file
  # this is just a precaution since simConfig pkl file has MOST of the info; ideally should adjust simConfig to contain
  # ALL of the required info
  from utils import backupcfg, safemkdir
  backupcfg(dconf['sim']['name'])
  safemkdir('data') # make sure data (output) directory exists

sim.net.createPops()                      # instantiate network populations
sim.net.createCells()                     # instantiate network cells based on defined populations
sim.net.connectCells()                    # create connections between cells based on params
sim.net.addStims()                      #instantiate netStim
sim.setupRecording()                  # setup variables to record for each cell (spikes, V traces, etc)

dSTDPmech = getAllSTDPObjects(sim) # get all the STDP objects up-front

if dconf['net']['homPlast']['On']:
  # call this once before the simulation
  initTargetFR(sim,dconf['net']['homPlast']['mintargetFR'],dconf['net']['homPlast']['maxtargetFR']) 
  initTargetW(sim,list(dconf['net']['homPlast']['mintargetFR'].keys()),synType=dconf['net']['homPlast']['synType']) # call this once before running the simulation.

def updateSTDPWeights (sim, W):
  #this function assign weights stored in 'ResumeSimFromFile' to all connections by matching pre and post neuron ids  
  # get all the simulation's cells (on a given node)
  for cell in sim.net.cells:
    cpostID = cell.gid#find postID
    WPost = W[(W.postid==cpostID)] #find the record for a connection with post neuron ID
    for conn in cell.conns:
      if 'hSTDP' not in conn: continue
      cpreID = conn.preGid  #find preID
      if type(cpreID) != int: continue
      cConnW = WPost[(WPost.preid==cpreID)] #find the record for a connection with pre and post neuron ID
      #find weight for the STDP connection between preID and postID
      for idx in cConnW.index:
        cW = cConnW.at[idx,'weight']
        conn['hObj'].weight[0] = cW
        hSTDP = conn.get('hSTDP')
        #hSTDP.cumreward = cConnW.at[idx,'cumreward']
        if dconf['verbose'] > 1: print('weight updated:', cW)
        
#if specified 'ResumeSim' = 1, load the connection data from 'ResumeSimFromFile' and assign weights to STDP synapses  
if dconf['simtype']['ResumeSim']:
  try:
    from simdat import readweightsfile2pdf
    A = readweightsfile2pdf(dconf['simtype']['ResumeSimFromFile'])
    updateSTDPWeights(sim, A[A.time == max(A.time)]) # take the latest weights saved
    if sim.rank==0: print('Updated STDP weights')
    if 'normalizeWeightsAtStart' in dconf['sim']:
      if dconf['sim']['normalizeWeightsAtStart']:
        print('normalizing adjustable weights at start')
        normalizeAdjustableWeights(sim, 0, lrecpop)
  except:
    print('Could not restore STDP weights from file.')

def setdminID (sim, lpop):
  # setup min ID for each population in lpop
  alltags = sim._gatherAllCellTags() #gather cell tags; see https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/sim/gather.py
  dGIDs = {pop:[] for pop in lpop}
  for tinds in range(len(alltags)):
    if alltags[tinds]['pop'] in lpop:
      dGIDs[alltags[tinds]['pop']].append(tinds)
  sim.simData['dminID'] = {pop:np.amin(dGIDs[pop]) for pop in lpop if len(dGIDs[pop])>0}

setdminID(sim, allpops)
tPerPlay = tstepPerAction*dconf['actionsPerPlay']
InitializeInputRates()
dsumWInit = getSumAdjustableWeights(sim) # get sum of adjustable weights at start of sim
sim.runSimWithIntervalFunc(tPerPlay,trainAgent) # has periodic callback to adjust STDP weights based on RL signal
if sim.rank==0 and fid4 is not None: fid4.close()
sim.gatherData() # gather data from different nodes
sim.saveData() # save data to disk

def LSynWeightToD (L):
  # convert list of synaptic weights to dictionary to save disk space
  print('converting synaptic weight list to dictionary...')
  dout = {}; doutfinal = {}
  for row in L:
    #t,preID,poID,w,cumreward = row
    t,preID,poID,w = row
    if preID not in dout:
      dout[preID] = {}
      doutfinal[preID] = {}
    if poID not in dout[preID]:
      dout[preID][poID] = []
      doutfinal[preID][poID] = []
    #dout[preID][poID].append([t,w,cumreward])
    dout[preID][poID].append([t,w])
  for preID in doutfinal.keys():
    for poID in doutfinal[preID].keys():
      doutfinal[preID][poID].append(dout[preID][poID][-1])
  return dout, doutfinal

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
      print(fn,'len(lw)=',len(lw),type(lw))
      os.unlink(fn) # remove the temporary file
      L = L + lw # concatenate to the list L
    #pickle.dump(L,open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb')) # this would save as a List
    # now convert the list to a dictionary to save space, and save it to disk
    dout, doutfinal = LSynWeightToD(L)
    pickle.dump(dout,open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb'))
    pickle.dump(doutfinal,open('data/'+dconf['sim']['name']+'synWeights_final.pkl', 'wb'))        

if sim.saveWeights: saveSynWeights()

def saveMotionFields (ldflow): pickle.dump(ldflow, open('data/'+dconf['sim']['name']+'MotionFields.pkl', 'wb'))

def saveObjPos (dobjpos):
  # save object position dictionary
  for k in dobjpos.keys(): dobjpos[k] = np.array(dobjpos[k])
  pickle.dump(dobjpos, open('data/'+dconf['sim']['name']+'objpos.pkl', 'wb'))

def saveAssignedFiringRates (dAllFiringRates): pickle.dump(dAllFiringRates, open('data/'+dconf['sim']['name']+'AssignedFiringRates.pkl', 'wb'))

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
  if sim.saveObjPos: saveObjPos(sim.AIGame.dObjPos)
  if sim.saveAssignedFiringRates: saveAssignedFiringRates(sim.AIGame.dAllFiringRates)
  if dconf['sim']['doquit']: quit()
