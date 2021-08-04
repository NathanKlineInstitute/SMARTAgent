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
from cells import intf7

random.seed(1234) # this will not work properly across runs with different number of nodes

sim.davgW = {} # average adjustable weights on a target population
sim.allTimes = []
sim.allRewards = [] # list to store all rewards
sim.allActions = [] # list to store all actions
sim.allProposedActions = [] # list to store all proposed actions
sim.allHits = [] #list to store all hits
sim.allMotorOutputs = [] # list to store firing rate of output motor neurons.
sim.followTargetSign = [] # whether racket moved closer (1) or farther (-1) from y intercept target at each step
sim.ActionsRewardsfilename = 'data/'+dconf['sim']['name']+'ActionsRewards.txt'
sim.MotorOutputsfilename = 'data/'+dconf['sim']['name']+'MotorOutputs.txt'
sim.WeightsRecordingTimes = []
sim.allRLWeights = [] # list to store weights --- should remove that
sim.allNonRLWeights = [] # list to store weights --- should remove that
sim.topologicalConns = dict() # dictionary to save topological connections.
sim.lastMove = dconf['moves']['NOMOVE']
#sim.NonRLweightsfilename = 'data/'+dconf['sim']['name']+'NonRLweights.txt'  # file to store weights
sim.plotWeights = 0  # plot weights
sim.saveWeights = 1  # save weights
if 'saveWeights' in dconf['sim']: sim.saveWeights = dconf['sim']['saveWeights']
sim.saveInputImages = 1 #save Input Images (5 game frames)
sim.saveMotionFields = dconf['sim']['saveMotionFields'] # whether to save the motion fields
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
EVPops = dconf['net']['EVPops'] # excitatory visual populations
EVDirPops = dconf['net']['EVDirPops'] # excitatory visual dir selective populations (corresponds to VD in cmat)
EVLocPops = dconf['net']['EVLocPops'] # excitatory visual location selective populations (corresponds to VL in cmat)
cmat = dconf['net']['cmat'] # connection matrix (for classes, synapses, probabilities [probabilities not used for topological conn])

dnumc = OrderedDict({ty:dconf['net']['allpops'][ty]*scale for ty in allpops}) # number of neurons of a given type

def getpadding ():
  # get padding-related parameters -- NB: not used
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

def setlrecpop ():
  lrecpop = ['EMUP', 'EMDOWN'] # which populations to record from
  if cmat['VD']['VD']['conv'] > 0 or \
     cmat['VD']['VL']['conv'] > 0 or \
     cmat['VL']['VL']['conv'] > 0 or \
     cmat['VL']['VD']['conv'] > 0 or \
     dconf['net']['VisualFeedback']:
    for pop in EVPops:
      lrecpop.append(pop)
    if dconf['net']['VisualFeedback'] and dnumc['ER']>0: lrecpop.append('ER')
  if dnumc['EA']>0 and (dconf['net']['RLconns']['RecurrentANeurons'] or \
                        dconf['net']['STDPconns']['RecurrentANeurons'] or \
                        dconf['net']['RLconns']['FeedbackMtoA'] or \
                        dconf['net']['STDPconns']['FeedbackMtoA']):
    lrecpop.append('EA')
  if dnumc['EA2']>0 and (dconf['net']['RLconns']['RecurrentA2Neurons'] or \
                         dconf['net']['STDPconns']['RecurrentA2Neurons'] or \
                         dconf['net']['RLconns']['FeedbackMtoA2'] or \
                         dconf['net']['STDPconns']['FeedbackMtoA2']): 
    lrecpop.append('EA2')
  if dconf['net']['RLconns']['Visual'] or dconf['net']['STDPconns']['Visual']:
    if lrecpop.count('EV4')==0: lrecpop.append('EV4')
    if lrecpop.count('EMT')==0: lrecpop.append('EMT')
  recITypes = False
  if dconf['net']['RLconns']['EIPlast'] or dconf['net']['STDPconns']['EIPlast']: recITypes = True
  elif 'Noise' in dconf['net']['RLconns']:
    if dconf['net']['RLconns']['Noise']:
      recITypes = True  
  if recITypes:
    for IType in ITypes:
      if dnumc[IType] > 0: lrecpop.append(IType)
  # this is not needed - since lrecpop has the postsynaptic type
  #if dnumc['EN'] > 0 and 'Noise' in dconf['net']['RLconns']:
  #  if dconf['net']['RLconns']['Noise']: lrecpop.append('EN')
  return lrecpop

lrecpop = setlrecpop()
        
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
simConfig.recordCellsSpikes = [-1] # this means record from all neurons - including stim populations, if any
simConfig.recordStep = dconf['sim']['recordStep'] # Step size in ms to save data (e.g. V traces, LFP, etc)
simConfig.filename = 'data/'+dconf['sim']['name']+'simConfig'  # Set file output name
simConfig.saveJson = False
simConfig.savePickle = True            # Save params, network and sim output to pickle file
simConfig.saveMat = False
simConfig.saveFolder = 'data'
# simConfig.backupCfg = ['sim.json', 'backupcfg/'+dconf['sim']['name']+'sim.json']
simConfig.createNEURONObj = True  # create HOC objects when instantiating network
simConfig.createPyStruct = True  # create Python structure (simulator-independent) when instantiating network
simConfig.analysis['plotTraces'] = {'include': [(pop, 0) for pop in ['ER','IR','EV1','EV1DE','ID','IV1','EV4','IV4','EMT','IMT','EMDOWN','EMUP','IM','IML','IMUP','IMDOWN','EA','IA','IAL','EA2','IA2','IA2L','EN']]}
simConfig.analysis['plotRaster'] = {'popRates':'overlay','showFig':dconf['sim']['doplot']}
#simConfig.analysis['plot2Dnet'] = True 
#simConfig.analysis['plotConn'] = True           # plot connectivity matrix
# simConfig.coreneuron = True
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
cfg.delayMinDend = dconf['net']['delayMinDend']
cfg.delayMaxDend = dconf['net']['delayMaxDend']
cfg.delayMinSoma = dconf['net']['delayMinSoma']
cfg.delayMaxSoma = dconf['net']['delayMaxSoma']

def getInitWeight (weight):
  """get initial weight for a connection
     checks if weightVar is non-zero, if so will use a uniform distribution 
     with range on interval: (1-var)*weight, (1+var)*weight
  """
  if cfg.weightVar == 0.0:
    return weight
  elif weight <= 0.0:
    return 0.0
  else:
    # print('uniform(%g,%g)' % (weight*(1.0-cfg.weightVar),weight*(1.0+cfg.weightVar)))
    return 'uniform(%g,%g)' % (max(0,weight*(1.0-cfg.weightVar)),weight*(1.0+cfg.weightVar))

def getCompFromSy (sy):
  if sy.count('2') > 0: return 'Dend'
  return 'Soma'
  
def getInitDelay (cmp='Dend'):
  a,b = float(dconf['net']['delayMin'+cmp]), float(dconf['net']['delayMax'+cmp])
  if a==b:
    return a
  else:
    return 'uniform(%g,%g)' % (a,b)
  
ECellModel = dconf['net']['ECellModel']
ICellModel = dconf['net']['ICellModel']

def getComp (sy):
  if ECellModel == 'INTF7' or ICellModel == 'INTF7':
    if sy.count('2') > 0:
      return 'Dend'
    return 'Soma'
  else:
    if sy.count('AM') or sy.count('NM'): return 'Dend'
    return 'Soma'

#Population parameters
for ty in allpops:
  if ty in ETypes:
    netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': ECellModel}
  else:
    netParams.popParams[ty] = {'cellType':ty, 'numCells': dnumc[ty], 'cellModel': ICellModel}

def makeECellModel (ECellModel):
  # create rules for excitatory neuron models
  EExcitSec = 'dend' # section where excitatory synapses placed
  PlastWeightIndex = 0 # NetCon weight index where plasticity occurs
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
    netParams.defaultThreshold = 0.0 
    for ty in ETypes:
      #netParams.popParams[ty]={'cellType':ty,'numCells':dnumc[ty],'cellModel':ECellModel}#, 'params':{'taue':5.35,'taui1':9.1,'taui2':0.07,'taum':20}}
      netParams.popParams[ty] = {'cellType':ty, 'cellModel': 'IntFire4', 'numCells': dnumc[ty], 'taue': 1.0}  # pop of IntFire4
  elif ECellModel == 'INTF7':
    EExcitSec = 'soma' # section where excitatory synapses placed
    simConfig.recordTraces = {'V_soma':{'var':'Vm'}}  # Dict with traces to record
    netParams.defaultThreshold = -40.0
    for ty in ETypes:
      netParams.popParams[ty] = {'cellType':ty, 'cellModel': 'INTF7', 'numCells': dnumc[ty]} # pop of IntFire4
      for k,v in intf7.INTF7E.dparam.items(): netParams.popParams[ty][k] = v
    PlastWeightIndex = intf7.dsyn['AM2']
  elif ECellModel == 'Friesen':
    cellRule = netParams.importCellParams(label='PYR_Friesen_rule', conds={'cellType': ETypes, 'cellModel': 'Friesen'},
                fileName='cells/friesen.py', cellName='MakeRSFCELL')
    cellRule['secs']['axon']['spikeGenLoc'] = 0.5  # spike generator location.
    EExcitSec = 'dend' # section where excitatory synapses placed
  elif ECellModel == 'HH':
    EExcitSec = 'soma'
    netParams.importCellParams(label='HHE_rule', conds={'cellType': ETypes}, fileName='cells/hht.py', cellName='HHE')
    netParams.cellParams['HHE_rule']['secs']['soma']['threshold'] = -10.0    
  return EExcitSec,PlastWeightIndex

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
    netParams.defaultThreshold = 0.0     
    for ty in ITypes:
      netParams.popParams[ty] = {'cellType':ty, 'cellModel': 'IntFire4', 'numCells': dnumc[ty], 'taue': 1.0}  # pop of IntFire4
  elif ICellModel == 'INTF7':
    EExcitSec = 'soma' # section where excitatory synapses placed
    simConfig.recordTraces = {'V_soma':{'var':'Vm'}}  # Dict with traces to record
    netParams.defaultThreshold = -40.0
    for ty in ITypes:
      netParams.popParams[ty] = {'cellType':ty, 'cellModel': 'INTF7', 'numCells': dnumc[ty]}
      if ty.count('L') > 0: # LTS
        for k,v in intf7.INTF7IL.dparam.items(): netParams.popParams[ty][k] = v
      else: # FS
        for k,v in intf7.INTF7I.dparam.items(): netParams.popParams[ty][k] = v          
  elif ICellModel == 'Friesen':
    cellRule = netParams.importCellParams(label='Bas_Friesen_rule', conds={'cellType': ITypes, 'cellModel': 'Friesen'},
                fileName='cells/friesen.py', cellName='MakeFSFCELL')
    cellRule['secs']['axon']['spikeGenLoc'] = 0.5  # spike generator location.
  elif ICellModel == 'HH':
    netParams.importCellParams(label='HHI_rule', conds={'cellType': ITypes}, fileName='cells/hht.py', cellName='HHI')
    netParams.cellParams['HHI_rule']['secs']['soma']['threshold'] = -10.0
      
EExcitSec,PlastWeightIndex = makeECellModel(ECellModel)
print('EExcitSec,PlastWeightIndex:',EExcitSec,PlastWeightIndex)
makeICellModel(ICellModel)
  
## Synaptic mechanism parameters
# note that these synaptic mechanisms are not used for the INTF7 neurons
# excitatory synaptic mechanism
netParams.synMechParams['AM2'] = netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}  
netParams.synMechParams['NM2'] = netParams.synMechParams['NMDA'] = {'mod': 'Exp2Syn', 'tau1': 0.15, 'tau2': 166.0, 'e': 0} # NMDA
# inhibitory synaptic mechanism
netParams.synMechParams['GA'] = netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80}

def readSTDPParams ():
  dSTDPparamsRL = {} # STDP-RL parameters for AMPA,NMDA synapses; generally uses shorter/longer eligibility traces
  lsy = ['AMPA', 'NMDA']
  if 'AMPAI' in dconf['RL']: lsy.append('AMPAI')
  if 'AMPAN' in dconf['RL']: lsy.append('AMPAN') # RL for NOISE synapses 
  for sy,gain in zip(lsy,[cfg.EEGain,cfg.EEGain,cfg.EIGain,cfg.EEGain]):
    dSTDPparamsRL[sy] = dconf['RL'][sy]
    for k in dSTDPparamsRL[sy].keys():
      if k.count('wt') or k.count('wbase') or k.count('wmax'): dSTDPparamsRL[sy][k] *= gain      
  lsy = ['AMPA', 'NMDA']
  if 'AMPAI' in dconf['STDP']: lsy.append('AMPAI')  
  dSTDPparams = {} # STDPL parameters for AMPA,NMDA synapses; generally uses shorter/longer eligibility traces
  for sy,gain in zip(lsy,[cfg.EEGain,cfg.EEGain,cfg.EIGain]):
    dSTDPparams[sy] = dconf['STDP'][sy]
    for k in dSTDPparams[sy].keys():
      if k.count('wt') or k.count('wbase') or k.count('wmax'): dSTDPparams[sy][k] *= gain
  dSTDPparamsRL['AM2']=dSTDPparamsRL['AMPA']; dSTDPparamsRL['NM2']=dSTDPparamsRL['NMDA']
  dSTDPparams['AM2']=dSTDPparams['AMPA']; dSTDPparams['NM2']=dSTDPparams['NMDA']    
  return dSTDPparamsRL, dSTDPparams
  
dSTDPparamsRL, dSTDPparams = readSTDPParams()

def getWeightIndex (synmech, cellModel):
  # get weight index for connParams
  if cellModel == 'INTF7': return intf7.dsyn[synmech]
  return 0
  
def setupStimMod ():
  # setup variable rate NetStim sources (send spikes based on image contents)
  lstimty = []
  inputPop = 'EV1' # which population gets the direct visual inputs (pixels)
  if dnumc['ER']>0: inputPop = 'ER'
  stimModLocW = dconf['net']['stimModVL']
  stimModDirW = dconf['net']['stimModVD']    
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    lpoty = [inputPop]
    for poty in ['EV1D'+Dir for Dir in ['E','NE','N', 'NW','W','SW','S','SE']]: lpoty.append(poty)
    wt = stimModLocW      
    for poty in lpoty:
      if dnumc[poty] <= 0: continue
      stimty = 'stimMod'+poty
      lstimty.append(stimty)
      netParams.popParams[stimty] = {'cellModel': 'NSLOC', 'numCells': dnumc[poty],'rate': 'variable', 'noise': 0, 'start': 0}
      blist = [[i,i] for i in range(dnumc[poty])]
      netParams.connParams[stimty+'->'+poty] = {
        'preConds': {'pop':stimty},
        'postConds': {'pop':poty},
        'weight':wt,
        'delay': getInitDelay('STIMMOD'),
        'connList':blist, 'weightIndex':getWeightIndex('AMPA',ECellModel)}
      wt = stimModDirW # rest of inputs use this weight
  else:
    # these are the image-based inputs provided to the R (retinal) cells
    netParams.stimSourceParams['stimMod'] = {'type': 'NetStim', 'rate': 'variable', 'noise': 0}  
    netParams.stimTargetParams['stimMod->'+inputPop] = {
      'source': 'stimMod',
      'conds': {'pop': inputPop},
      'convergence': 1,
      'weight': stimModLocW,
      'delay': 1,
      'synMech': 'AMPA'}
    for pop in ['EV1D'+Dir for Dir in ['E','NE','N', 'NW','W','SW','S','SE']]:
      netParams.stimTargetParams['stimMod->'+pop] = {
        'source': 'stimMod',
        'conds': {'pop': pop},
        'convergence': 1,
        'weight': stimModDirW,
        'delay': 1,
        'synMech': 'AMPA'}
  return lstimty

sim.lstimty = setupStimMod() # when using IntFire4 cells lstimty has the NetStim populations that send spikes to EV1, EV1DE, etc.
for ty in sim.lstimty: allpops.append(ty)
            
# Stimulation parameters
def setupNoiseStim ():
  lnoisety = []
  dnoise = dconf['noise']
  # setup noisy NetStim sources (send random spikes)
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    lpoty = dnoise.keys()
    for poty in lpoty:
      lsy = dnoise[poty].keys()
      for sy in lsy:
        Weight,Rate = dnoise[poty][sy]['w'],dnoise[poty][sy]['rate']
        if Weight > 0.0 and Rate > 0.0: # only create the netstims if rate,weight > 0
          stimty = 'stimNoise'+poty+'_'+sy
          netParams.popParams[stimty] = {'cellModel': 'NetStim', 'numCells': dnumc[poty],'rate': Rate, 'noise': 1.0, 'start': 0}
          blist = [[i,i] for i in range(dnumc[poty])]
          netParams.connParams[stimty+'->'+poty] = {
            'preConds': {'pop':stimty},
            'postConds': {'pop':poty},
            'weight':Weight,
            'delay': getInitDelay(getCompFromSy(sy)),
            'connList':blist,
            'weightIndex':getWeightIndex(sy,ECellModel)}
          lnoisety.append(stimty)
  else:
    # setup noise inputs
    lpoty = dnoise.keys()
    for poty in lpoty:
      lsy = dnoise[poty].keys()
      for sy in lsy:
        Weight,Rate = dnoise[poty][sy]['w'],dnoise[poty][sy]['rate']
        if Weight > 0.0 and Rate > 0.0: # only create the netstims if rate,weight > 0
          stimty = poty+'Mbkg'+sy
          netParams.stimSourceParams[stimty] = {'type': 'NetStim', 'rate': Rate, 'noise': 1.0}
          netParams.stimTargetParams[poty+'Mbkg->all'] = {
            'source': stimty, 'conds': {'cellType': EMotorPops}, 'weight': Weight, 'delay': 'max(1, normal(5,2))', 'synMech': sy}
          # lnoisety.append(ty+'Mbkg'+sy)
  return lnoisety

sim.lnoisety = setupNoiseStim()
for ty in sim.lnoisety: allpops.append(ty)
      
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
    
#Local excitation
#E to E recurrent connectivity within visual areas
for epop in EVPops:
  if dnumc[epop] <= 0: continue # skip rule setup for empty population
  prety = poty = epop
  repstr = 'VD' # replacement presynaptic type string (VD -> EV1DE, EV1DNE, etc.; VL -> EV1, EV4, etc.)  
  if prety in EVLocPops: repstr = 'VL'
  wAM, wNM = cmat[repstr][repstr]['AM2'], cmat[repstr][repstr]['NM2']  
  for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[wAM*cfg.EEGain, wNM*cfg.EEGain]):
    k = strty+prety+'->'+strty+poty
    if weight <= 0.0: continue
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat, repstr, repstr, dnumc[prety]),
      'weight': getInitWeight(weight),
      'delay': getInitDelay('Dend'),
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5,
      'weightIndex':getWeightIndex(synmech, ECellModel)
    }            
    useRL = useSTDP = False
    if prety in EVDirPops:
      if dconf['net']['RLconns']['RecurrentDirNeurons']: useRL = True
      if dconf['net']['STDPconns']['RecurrentDirNeurons']: useSTDP = True
    if prety in EVLocPops:
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
          'weight': cmat['ER']['IR']['AM2'] * cfg.EIGain,
          'delay': getInitDelay('Dend'),
          'synMech': 'AMPA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('AM2', ICellModel)}
  if VTopoI and dconf['sim']['useReducedNetwork']==0: netParams.connParams['ER->IR']['connList'] = blistERtoIR
  else: netParams.connParams['ER->IR']['convergence'] = getconv(cmat, 'ER', 'IR', dnumc['ER']) 
  
netParams.connParams['EV1->IV1'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'IV1'},
        'weight': cmat['EV1']['IV1']['AM2'] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('AM2', ICellModel)}

if VTopoI and dconf['sim']['useReducedNetwork']==0:
  netParams.connParams['EV1->IV1']['connList'] = blistEV1toIV1
  sim.topologicalConns['EV1->IV1'] = {'blist':blistEV1toIV1, 'coords':connCoordsEV1toIV1}
else:
  netParams.connParams['EV1->IV1']['convergence'] = getconv(cmat, 'EV1', 'IV1', dnumc['EV1'])

if dnumc['ID']>0:
  EVDirPops = dconf['net']['EVDirPops']
  IVDirPops = dconf['net']['IVDirPops']
  for prety in EVDirPops:
    for poty in IVDirPops:
      netParams.connParams[prety+'->'+poty] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat, 'VD', 'ID', dnumc[prety]),
        'weight': cmat['VD']['ID']['AM2'] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex('AM2', ICellModel)}

netParams.connParams['EV4->IV4'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'IV4'},
        'weight': cmat['EV4']['IV4']['AM2'] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex('AM2', ICellModel)}
if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['EV4->IV4']['connList'] = blistEV4toIV4
  sim.topologicalConns['EV4->IV4'] = {'blist':blistEV4toIV4, 'coords':connCoordsEV4toIV4}
else: 
  netParams.connParams['EV4->IV4']['convergence'] = getconv(cmat,'EV4','IV4', dnumc['EV4'])

netParams.connParams['EMT->IMT'] = {
        'preConds': {'pop': 'EMT'},
        'postConds': {'pop': 'IMT'},
        'weight': cmat['EMT']['IMT']['AM2'] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('AM2', ICellModel)}
if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['EMT->IMT']['connList'] = blistEMTtoIMT
  sim.topologicalConns['EMT->IMT'] = {'blist':blistEMTtoIMT, 'coords':connCoordsEMTtoIMT}
else: 
  netParams.connParams['EMT->IMT']['convergence'] = getconv(cmat, 'EMT', 'IMT', dnumc['EMT'])

for prety,poty in zip(['EA','EA','EA2','EA2'],['IA','IAL','IA2','IA2L']):
  if dnumc[prety] <= 0 or dnumc[poty] <= 0: continue
  for sy in ['AM2','NM2']:
    if sy not in cmat[prety][poty]: continue
    k = prety+'->'+poty+sy
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat, prety, poty, dnumc[prety]),
      'weight': cmat[prety][poty][sy] * cfg.EIGain,
      'delay': getInitDelay('Dend'),
      'synMech': sy, 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex(sy, ICellModel)}
    if sy.count('AM') > 0:
      if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
        netParams.connParams[k]['weight'] = getInitWeight(cmat[prety][poty]['AM2'] * cfg.EIGain)
      elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
        netParams.connParams[k]['weight'] = getInitWeight(cmat[prety][poty]['AM2'] * cfg.EIGain)    
 
for prety in EMotorPops:
  if dnumc[prety] <= 0: continue
  for poty in ['IM', 'IML']:
    if dnumc[poty] <= 0: continue
    for sy in ['AM2','NM2']:
      if sy not in cmat['EM'][poty]: continue
      k = prety+'->'+poty+sy
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat, 'EM', poty, dnumc[prety]),
        'weight': cmat['EM'][poty][sy] * cfg.EIGain,
        'delay': getInitDelay('Dend'),
        'synMech': sy, 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex(sy, ICellModel)}
      if sy.count('AM') > 0:
        if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
          netParams.connParams[k]['weight'] = getInitWeight(cmat['EM'][poty]['AM2'] * cfg.EIGain)
        elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
          netParams.connParams[k]['weight'] = getInitWeight(cmat['EM'][poty]['AM2'] * cfg.EIGain)    
   
# reciprocal inhibition - only active when all relevant populations created - not usually used
for prety in EMotorPops:
  for epoty in EMotorPops:
    if epoty == prety: continue # no self inhib here
    poty = 'IM' + epoty[2:] # change name to interneuron
    k = prety + '->' + poty
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat, 'EM', 'IRecip', dnumc[prety]),
      'weight': cmat['EM']['IRecip']['AM2'] * cfg.EIGain,
      'delay': getInitDelay('Dend'),
      'synMech': 'AMPA', 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex('AM2', ICellModel)}
    if dconf['net']['RLconns']['EIPlast'] and dSTDPparamsRL['AMPAI']['RLon']: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAI']}
      netParams.connParams[k]['weight'] = getInitWeight(cmat['EM']['IRecip']['AM2'] * cfg.EIGain)      
    elif dconf['net']['STDPconns']['EIPlast'] and dSTDPparams['AMPAI']['STDPon']:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams['AMPAI']}
      netParams.connParams[k]['weight'] = getInitWeight(cmat['EM']['IRecip']['AM2'] * cfg.EIGain)            
    
#Local inhibition
#I to E within area
if dnumc['ER']>0:
  netParams.connParams['IR->ER'] = {
          'preConds': {'pop': 'IR'},
          'postConds': {'pop': 'ER'},
          'weight': cmat['IR']['ER']['GA'] * cfg.IEGain,
          'delay': getInitDelay('Soma'),
          'synMech': 'GABA', 'sec':'soma', 'loc':0.5, 'weightIndex':getWeightIndex('GA', ICellModel)}  
  if VTopoI and dconf['sim']['useReducedNetwork']==0: 
    netParams.connParams['IR->ER']['connList'] = blistIRtoER
    sim.topologicalConns['IR->ER'] = {'blist':blistIRtoER, 'coords':connCoordsIRtoER}
  else: 
    netParams.connParams['IR->ER']['convergence'] = getconv(cmat, 'IR', 'ER', dnumc['IR'])
  
netParams.connParams['IV1->EV1'] = {
  'preConds': {'pop': 'IV1'},
  'postConds': {'pop': 'EV1'},
  'weight': cmat['IV1']['EV1']['GA'] * cfg.IEGain,
  'delay': getInitDelay('Soma'),
  'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['IV1->EV1']['connList'] = blistIV1toEV1
  sim.topologicalConns['IV1->EV1'] = {'blist':blistIV1toEV1, 'coords':connCoordsIV1toEV1}
else: 
  netParams.connParams['IV1->EV1']['convergence'] = getconv(cmat, 'IV1', 'EV1', dnumc['IV1'])  

if dnumc['ID']>0:
  IVDirPops = dconf['net']['IVDirPops']
  for prety in IVDirPops:
    for poty in EVDirPops:
      netParams.connParams[prety+'->'+poty] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat, 'ID', 'ED', dnumc['ID']),
        'weight': cmat['ID']['ED']['GA'] * cfg.IEGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}

netParams.connParams['IV4->EV4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'EV4'},
        'weight': cmat['IV4']['EV4']['GA'] * cfg.IEGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['IV4->EV4']['connList'] = blistIV4toEV4
  sim.topologicalConns['IV4->EV4'] = {'blist':blistIV4toEV4, 'coords':connCoordsIV4toEV4}
else: 
  netParams.connParams['IV4->EV4']['convergence'] = getconv(cmat,'IV4','EV4', dnumc['IV4'])

netParams.connParams['IMT->EMT'] = {
        'preConds': {'pop': 'IMT'},
        'postConds': {'pop': 'EMT'},
        'weight': cmat['IMT']['EMT']['GA'] * cfg.IEGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
if VTopoI and dconf['sim']['useReducedNetwork']==0: 
  netParams.connParams['IMT->EMT']['connList'] = blistIMTtoEMT
  sim.topologicalConns['IMT->EMT'] = {'blist':blistIMTtoEMT, 'coords':connCoordsIMTtoEMT}
else: 
  netParams.connParams['IMT->EMT']['convergence'] = getconv(cmat,'IMT','EMT',dnumc['IMT'])

# I -> E for motor populations
for prety,sy in zip(['IM', 'IML'],['GA','GA2']):
  for poty in EMotorPops: 
    netParams.connParams[prety+'->'+poty] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat,prety,'EM', dnumc[prety]),
      'weight': cmat[prety]['EM'][sy] * cfg.IEGain,
      'delay': getInitDelay(getCompFromSy(sy)),
      'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex(sy, ECellModel)}

for prety,poty,sy in zip(['IA','IAL','IA2','IA2L'],['EA','EA','EA2','EA2'],['GA','GA2','GA','GA2']):  
  netParams.connParams[prety+'->'+poty] = {
    'preConds': {'pop': prety},
    'postConds': {'pop': poty},
    'convergence': getconv(cmat,prety,poty, dnumc[prety]),
    'weight': cmat[prety][poty][sy] * cfg.IEGain,
    'delay': getInitDelay(getCompFromSy(sy)),
    'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex(sy, ECellModel)}
  
#I to I
for preIType in ITypes:
  sy = 'GA'
  if preIType.count('L') > 0: sy = 'GA2'
  for poIType in ITypes:
    if preIType not in dnumc or poIType not in dnumc: continue
    if dnumc[preIType] <= 0 or dnumc[poIType] <= 0: continue
    if poIType not in cmat[preIType] or \
       getconv(cmat,preIType,poIType,dnumc[preIType])<=0 or \
       cmat[preIType][poIType][sy]<=0: continue
    netParams.connParams[preIType+'->'+poIType] = {
      'preConds': {'pop': preIType},
      'postConds': {'pop': poIType},
      'convergence': getconv(cmat,preIType,poIType,dnumc[preIType]),
      'weight': cmat[preIType][poIType][sy] * cfg.IIGain, 
      'delay': getInitDelay(getCompFromSy(sy)),
      'synMech': 'GABA', 'sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex(sy, ICellModel)}

#E to E feedforward connections - AMPA,NMDA
lprety,lpoty,lblist,lconnsCoords = [],[],[],[]
if not dconf['sim']['useReducedNetwork']:
  if dnumc['ER']>0:
    lprety.append('ER')
    lpoty.append('EV1')
    lblist.append(blistERtoEV1)
    lconnsCoords.append(connCoordsERtoEV1)
  lprety.append('EV1'); lpoty.append('EV4'); lblist.append(blistEV1toEV4); lconnsCoords.append(connCoordsEV1toEV4)
  lprety.append('EV4'); lpoty.append('EMT'); lblist.append(blistEV4toEMT); lconnsCoords.append(connCoordsEV4toEMT)
  for prety,poty,blist,connCoords in zip(lprety,lpoty,lblist,lconnsCoords):  
    for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat[prety][poty]['AM2']*cfg.EEGain,cmat[prety][poty]['NM2']*cfg.EEGain]):
      k = strty+prety+'->'+strty+poty
      if weight <= 0.0: continue        
      netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'weight': weight ,
            'delay': getInitDelay('Dend'),
            'synMech': synmech,'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)}
      if VTopoI: # topological connections
        netParams.connParams[k]['connList'] = blist
        sim.topologicalConns[prety+'->'+poty] = {}
        sim.topologicalConns[prety+'->'+poty]['blist'] = blist
        sim.topologicalConns[prety+'->'+poty]['coords'] = connCoords
      else: # random connectivity
        netParams.connParams[k]['convergence'] = getconv(cmat,prety,poty,dnumc[prety])
      if dconf['net']['RLconns']['Visual'] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['weight'] = getInitWeight(weight) # make sure non-uniform weights
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif dconf['net']['STDPconns']['Visual'] and dSTDPparams[synmech]['STDPon']:
        netParams.connParams[k]['weight'] = getInitWeight(weight) # make sure non-uniform weights
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

if dconf['net']['VisualFeedback'] and not dconf['sim']['useReducedNetwork']:
  # visual area feedback connections
  pretyList = ['EV1','EV4','EMT']
  potyList = ['ER','EV1','EV4']
  allconnList = [blistEV1toER,blistEV4toEV1,blistEMTtoEV4]
  allconnCoords = [connCoordsEV1toER,connCoordsEV4toEV1,connCoordsEMTtoEV4]
  for prety,poty,connList,connCoords in zip(pretyList,potyList,allconnList,allconnCoords):
    if dnumc[prety] <= 0 or dnumc[poty] <= 0: continue # skip empty pops
    for strty,synmech,synweight in zip(['','n'],['AM2', 'NM2'],[cmat[prety][poty]['AM2']*cfg.EEGain, cmat[prety][poty]['NM2']*cfg.EEGain]):
        if synweight <= 0.0: continue
        k = strty+prety+'->'+strty+poty
        netParams.connParams[k] = {
          'preConds': {'pop': prety},
          'postConds': {'pop': poty},
          'connList': connList,
          'weight': getInitWeight(synweight),
          'delay': getInitDelay('Dend'),
          'synMech': synmech,'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)} # 'weight' should be fixed
        sim.topologicalConns[prety+'->'+poty] = {}
        sim.topologicalConns[prety+'->'+poty]['blist'] = connList
        sim.topologicalConns[prety+'->'+poty]['coords'] = connCoords
        # only turn on plasticity when specified to do so
        if dconf['net']['RLconns']['FeedbackLocNeurons'] and dSTDPparamsRL[synmech]['RLon']: 
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
        elif dconf['net']['STDPconns']['FeedbackLocNeurons'] and dSTDPparams[synmech]['STDPon']:
          netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}
          
  #I to E feedback connections
  netParams.connParams['IV1->ER'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'ER'},
        'connList': blistIV1toER,
        'weight': cmat['IV1']['ER']['GA'] * cfg.IEGain, 
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA','sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
  sim.topologicalConns['IV1->ER'] = {}
  sim.topologicalConns['IV1->ER']['blist'] = blistIV1toER
  sim.topologicalConns['IV1->ER']['coords'] = connCoordsIV1toER
  netParams.connParams['IV4->EV1'] = {
          'preConds': {'pop': 'IV4'},
          'postConds': {'pop': 'EV1'},
          'connList': blistIV4toEV1,
          'weight': cmat['IV4']['EV1']['GA'] * cfg.IEGain, 
          'delay': getInitDelay('Soma'),
          'synMech': 'GABA','sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
  sim.topologicalConns['IV4->EV1'] = {}
  sim.topologicalConns['IV4->EV1']['blist'] = blistIV4toEV1
  sim.topologicalConns['IV4->EV1']['coords'] = connCoordsIV4toEV1
  netParams.connParams['IMT->EV4'] = {
          'preConds': {'pop': 'IMT'},
          'postConds': {'pop': 'EV4'},
          'connList': blistIMTtoEV4,
          'weight': cmat['IMT']['EV4']['GA'] * cfg.IEGain, 
          'delay': getInitDelay('Soma'),
          'synMech': 'GABA','sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ECellModel)}
  sim.topologicalConns['IMT->EV4'] = {'blist':blistIMTtoEV4, 'coords':connCoordsIMTtoEV4}

#I to I - between areas
if dconf['sim']['useReducedNetwork']==0:
  netParams.connParams['IV1->IV4'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistIV1toIV4,
        'weight': cmat['IV1']['IV4']['GA'] * cfg.IIGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA','sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ICellModel)}
  sim.topologicalConns['IV1->IV4'] = {'blist':blistIV1toIV4, 'coords':connCoordsIV1toIV4}

  netParams.connParams['IV4->IMT'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IMT'},
        'connList': blistIV4toIMT,
        'weight': cmat['IV4']['IMT']['GA'] * cfg.IIGain,
        'delay': getInitDelay('Soma'),
        'synMech': 'GABA','sec':'soma', 'loc':0.5,'weightIndex':getWeightIndex('GA', ICellModel)}
  sim.topologicalConns['IV4->IMT'] = {'blist':blistIV4toIMT, 'coords':connCoordsIV4toIMT}

def connectEVToTarget (lpoty, useTopological):  
  if dconf['sim']['useReducedNetwork']:    
    print(cLV1toEA)
    synmech = 'AM2'
    weight = cfg.EEGain*cmat['VL']['EA']['AM2']
    netParams.connParams['EV1->EA'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1toEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    if dconf['net']['RLconns']['FeedForwardLocNtoA'] and dSTDPparamsRL[synmech]['RLon']: 
      netParams.connParams['EV1->EA']['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
    elif dconf['net']['STDPconns']['FeedForwardLocNtoA'] and dSTDPparams[synmech]['STDPon']:
      netParams.connParams['EV1->EA']['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}
    weight = cfg.EEGain*cmat['VD']['EA']['AM2']
    netParams.connParams['EV1DE->EA'] = {
        'preConds': {'pop': 'EV1DE'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1DEtoEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    netParams.connParams['EV1DNE->EA'] = {
        'preConds': {'pop': 'EV1DNE'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1DNEtoEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    netParams.connParams['EV1DN->EA'] = {
        'preConds': {'pop': 'EV1DN'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1DNtoEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    netParams.connParams['EV1DNW->EA'] = {
        'preConds': {'pop': 'EV1DNW'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1DNWtoEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    netParams.connParams['EV1DW->EA'] = {
        'preConds': {'pop': 'EV1DW'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1DWtoEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    netParams.connParams['EV1DSW->EA'] = {
        'preConds': {'pop': 'EV1DSW'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1DSWtoEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    netParams.connParams['EV1DS->EA'] = {
        'preConds': {'pop': 'EV1DS'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1DStoEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    netParams.connParams['EV1DSE->EA'] = {
        'preConds': {'pop': 'EV1DSE'},
        'postConds': {'pop': 'EA'},
        'connList': cLV1DSEtoEA, 
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    ldirconns = ['EV1DE->EA','EV1DNE->EA','EV1DN->EA','EV1DNW->EA','EV1DW->EA','EV1DSW->EA','EV1DS->EA','EV1DSE->EA']
    if dconf['net']['RLconns']['FeedForwardDirNtoA'] and dSTDPparamsRL[synmech]['RLon']:
      for dirconns in ldirconns: 
        netParams.connParams[dirconns]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
    elif dconf['net']['STDPconns']['FeedForwardDirNtoA'] and dSTDPparams[synmech]['STDPon']:
      for dirconns in ldirconns:
        netParams.connParams[dirconns]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}    
  else:
    # connect excitatory visual area neurons to list of postsynaptic types (lpoty)
    for prety in EVPops:
      if dnumc[prety] <= 0: continue
      for poty in lpoty:
        if dnumc[poty] <= 0: continue
        suffix = 'M'      
        if poty == 'EA': suffix = 'A'
        if poty == 'EA2': suffix = 'A2'
        if useTopological:
          try: div = dconf['net']['alltopoldivcons'][prety][poty]
          except: div = 3          
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
          sim.topologicalConns[prety+'->'+poty] = {'blist':blist,'coords':connCoords}          
        repstr = 'VD' # replacement presynaptic type string (VD -> EV1DE, EV1DNE, etc.; VL -> EV1, EV4, etc.)
        if prety in EVLocPops: repstr = 'VL'
        lsynw = [cmat[repstr][poty]['AM2']*cfg.EEGain, cmat[repstr][poty]['NM2']*cfg.EEGain]          
        for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],lsynw):        
          k = strty+prety+'->'+strty+poty
          if weight <= 0.0: continue
          if poty == 'EA' or poty == 'EA2':
            wtval = weight # make sure EA,EA2 get enough input
          else:
            wtval = getInitWeight(weight)
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'weight': wtval,
            'delay': getInitDelay('Dend'),
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
          }
          if useTopological:
            netParams.connParams[k]['connList'] = blist
          else:
            netParams.connParams[k]['convergence'] = getconv(cmat,repstr,poty,dnumc[prety])
          useRL = useSTDP = False
          if prety in EVDirPops:
            if dconf['net']['RLconns']['FeedForwardDirNto'+suffix]: useRL = True
            if dconf['net']['STDPconns']['FeedForwardDirNto'+suffix]: useSTDP = True          
          if prety in EVLocPops:
            if dconf['net']['RLconns']['FeedForwardLocNto'+suffix]: useRL = True
            if dconf['net']['STDPconns']['FeedForwardLocNto'+suffix]: useSTDP = True
          if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif dSTDPparams[synmech]['STDPon'] and useSTDP:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}                           

connectEVToTarget(['EA','EA2'], dconf['architectureVtoA']['useTopological']) # connect primary visual to visual association
connectEVToTarget(EMotorPops, dconf['architectureVtoM']['useTopological']) # connect primary visual to motor

# add connections from first to second visual association area
# EA -> EA2 (feedforward)
prety = 'EA'; poty = 'EA2'
if dnumc[prety] > 0 and dnumc[poty] > 0:  
  lsynw = [cmat[prety][poty]['AM2']*cfg.EEGain, cmat[prety][poty]['NM2']*cfg.EEGain]
  for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],lsynw):
    k = strty+prety+'->'+strty+poty
    if weight <= 0.0: continue    
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat,prety,poty,dnumc[prety]),
      'weight': getInitWeight(weight),
      'delay': getInitDelay('Dend'),
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    useRL = useSTDP = False
    ffconnty = 'FeedForwardAtoA2'
    if dconf['net']['RLconns'][ffconnty]: useRL = True
    if dconf['net']['STDPconns'][ffconnty]: useSTDP = True          
    if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
    elif dSTDPparams[synmech]['STDPon'] and useSTDP:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

# EA2 -> EA (feedback)
prety = 'EA2'; poty = 'EA'
if dnumc[prety] > 0 and dnumc[poty] > 0:  
  lsynw = [cmat[prety][poty]['AM2']*cfg.EEGain, cmat[prety][poty]['NM2']*cfg.EEGain]
  for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],lsynw):
    k = strty+prety+'->'+strty+poty
    if weight <= 0.0: continue
    netParams.connParams[k] = {
      'preConds': {'pop': prety},
      'postConds': {'pop': poty},
      'convergence': getconv(cmat,prety,poty,dnumc[prety]),
      'weight': getInitWeight(weight),
      'delay': getInitDelay('Dend'),
      'synMech': synmech,
      'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
    }
    useRL = useSTDP = False
    fbconnty = 'FeedbackA2toA'
    if dconf['net']['RLconns'][fbconnty]: useRL = True
    if dconf['net']['STDPconns'][fbconnty]: useSTDP = True          
    if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
    elif dSTDPparams[synmech]['STDPon'] and useSTDP:
      netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

      
# Add connections from visual association areas to motor cortex, and reccurrent conn within visual association areas
for prety,ffconnty,recconnty in zip(['EA', 'EA2'],['FeedForwardAtoM','FeedForwardA2toM'],['RecurrentANeurons','RecurrentA2Neurons']):
  if dnumc[prety] <= 0: continue
  lsynw = [cmat[prety]['EM']['AM2']*cfg.EEGain, cmat[prety]['EM']['NM2']*cfg.EEGain]
  for poty in EMotorPops:
    if dnumc[poty] <= 0: continue
    for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],lsynw):
      k = strty+prety+'->'+strty+poty
      if weight <= 0.0: continue              
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat,prety,'EM', dnumc[prety]),
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
      }
      useRL = useSTDP = False
      if dconf['net']['RLconns'][ffconnty]: useRL = True
      if dconf['net']['STDPconns'][ffconnty]: useSTDP = True          
      if dSTDPparamsRL[synmech]['RLon'] and useRL: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif dSTDPparams[synmech]['STDPon'] and useSTDP:
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}                                      
  # add recurrent plastic connectivity within EA populations
  poty = prety
  if getconv(cmat,prety,poty,dnumc[prety])>0 and dnumc[poty]>0:
    for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat[prety][poty]['AM2']*cfg.EEGain, cmat[prety][poty]['NM2']*cfg.EEGain]):
      k = strty+prety+'->'+strty+poty
      if weight <= 0.0: continue              
      netParams.connParams[k] = {
        'preConds': {'pop': prety},
        'postConds': {'pop': poty},
        'convergence': getconv(cmat,prety,poty,dnumc[prety]),
        'weight': getInitWeight(weight),
        'delay': getInitDelay('Dend'),
        'synMech': synmech,
        'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
      }
      if dconf['net']['RLconns'][recconnty] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
      elif dconf['net']['STDPconns'][recconnty] and dSTDPparams[synmech]['STDPon']:
        netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}            
          
# add recurrent plastic connectivity within EM populations
if getconv(cmat,'EM','EM',dnumc['EMDOWN']) > 0:
  for prety in EMotorPops:
    for poty in EMotorPops:
      if prety==poty or dconf['net']['EEMRecProbCross']: # same types or allowing cross EM population connectivity
        for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat['EM']['EM']['AM2']*cfg.EEGain, cmat['EM']['EM']['NM2']*cfg.EEGain]):
          k = strty+prety+'->'+strty+poty
          if weight <= 0.0: continue                  
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': getconv(cmat,'EM','EM', dnumc[prety]),
            'weight': getInitWeight(weight),
            'delay': getInitDelay('Dend'),
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
          }
          if dconf['net']['RLconns']['RecurrentMNeurons'] and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif dconf['net']['STDPconns']['RecurrentMNeurons'] and dSTDPparams[synmech]['STDPon']:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}            

# add feedback plastic connectivity from EM populations to association populations
if getconv(cmat,'EM','EA',dnumc['EMDOWN'])>0:
  for prety in EMotorPops:
    for poty in ['EA','EA2']:
        for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat['EM'][poty]['AM2']*cfg.EEGain, cmat['EM'][poty]['NM2']*cfg.EEGain]):
          k = strty+prety+'->'+strty+poty
          if weight <= 0.0: continue
          netParams.connParams[k] = {
            'preConds': {'pop': prety},
            'postConds': {'pop': poty},
            'convergence': getconv(cmat,'EM',poty,dnumc[prety]),
            'weight': getInitWeight(weight),
            'delay': getInitDelay('Dend'),
            'synMech': synmech,
            'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
          }
          useRL = useSTDP = False
          if poty == 'EA':
            if dconf['net']['RLconns']['FeedbackMtoA']: useRL = True
            if dconf['net']['STDPconns']['FeedbackMtoA']: useSTDP = True
          elif poty == 'EA2':
            if dconf['net']['RLconns']['FeedbackMtoA2']: useRL = True
            if dconf['net']['STDPconns']['FeedbackMtoA2']: useSTDP = True                          
          if useRL and dSTDPparamsRL[synmech]['RLon']: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL[synmech]}
          elif useSTDP and dSTDPparams[synmech]['STDPon']:
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparams[synmech]}

def wireNoisePops ():
  if 'EN' not in dnumc: return
  prety = 'EN'
  if dnumc[prety] <= 0 or prety not in cmat: return # dynamic E Noise population
  lpoty = ETypes
  for ty in ITypes: lpoty.append(ty)
  for poty in lpoty:
    if dnumc[poty] <= 0: continue
    gn = cfg.EEGain
    if poty in ITypes: gn = cfg.EIGain
    # print(prety, poty, dnumc[prety], dnumc[poty])
    if getconv(cmat, prety, poty, dnumc[prety]) > 0:
      for strty,synmech,weight in zip(['','n'],['AM2', 'NM2'],[cmat[prety][poty]['AM2']*gn, cmat[prety][poty]['NM2']*gn]):
        k = strty+prety+'->'+strty+poty
        if weight <= 0.0: continue
        netParams.connParams[k] = {
          'preConds': {'pop': prety},
          'postConds': {'pop': poty},
          'convergence': getconv(cmat,prety,poty,dnumc[prety]),
          'weight': getInitWeight(weight),
          'delay': getInitDelay('Dend'),
          'synMech': synmech,
          'sec':EExcitSec, 'loc':0.5,'weightIndex':getWeightIndex(synmech, ECellModel)
        }
        if synmech.count('AM') > 0:
          if dconf['net']['RLconns']['Noise'] and dSTDPparamsRL['AMPAN']['RLon']: # only turn on plasticity when specified to do so
            netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': dSTDPparamsRL['AMPAN']}
            netParams.connParams[k]['weight'] = getInitWeight(cmat[prety][poty]['AM2'] * gn)
                    
if 'EN' in dnumc: wireNoisePops() # connect dynamic noise populations

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
        W += float(conn['hObj'].weight[PlastWeightIndex])
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
        W += float(conn['hObj'].weight[PlastWeightIndex])
        N += 1
  return W, N
  
def recordAdjustableWeightsPop (sim, t, popname):
  # record the plastic weights for specified popname
  lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids] # this is the set of popname cells
  for cell in lcell:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        #hstdp = conn.get('hSTDP')
        lsynweights.append([t,conn.preGid,cell.gid,float(conn['hObj'].weight[PlastWeightIndex])])#,hstdp.cumreward])
  return len(lcell)
                    
def recordAdjustableWeights (sim, t, lpop):
  """ record the STDP weights during the simulation - called in trainAgent
  """
  for pop in lpop: recordAdjustableWeightsPop(sim, t, pop)

    
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
          conn['hObj'].weight[PlastWeightIndex] *= dfctr[pop] 

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
                conn['hObj'].weight[PlastWeightIndex] *= fctrB
  else:
    davg = getAverageAdjustableWeights(sim, lpop)
    try:
      dfctr = {}
      MinFctr, MaxFctr = dconf['net']['EEMWghtThreshMin'], dconf['net']['EEMWghtThreshMax']      
      # normalize weights across populations to avoid bias    
      initavgW = cmat['EA']['EM']['AM2'] # initial average weight <<-- THAT ASSUMES ONLY USING NORM ON EM POPULATIONS!
      if dconf['net']['EEMPopNorm']:
        curravgW = np.mean([davg[k] for k in lpop]) # current average weight
        if curravgW <= 0.0: curravgw = initavgW
        if curravgW > 0 and curravgW != initavgW:
          fctrA = curravgW / initavgW
          dochange = False
          if fctrA < MinFctr:
            fctrB = (MinFctr * initavgW) / curravgW # factor to restore weights to boundary
            dochange = True
          elif fctrA > MaxFctr:            
            # fctrB = ((1.0+MaxFctr/2.0)*initavgW) / curravgW # factor to move weight to mid btwn start and max
            fctrB = initavgW / curravgW # go back to initial
            dochange = True
          # print('initW:',initW,'currW:',currW,'fctr:',fctr)
          if dochange:
            for k in lpop: dfctr[k] = fctrB
          else:
            for k in lpop: dfctr[k] = 1.0    
      else:
        MinFctr, MaxFctr = dconf['net']['EEMWghtThreshMin'], dconf['net']['EEMWghtThreshMax']
        for k in lpop:
          curravgW = davg[k]
          if curravgW > 0 and curravgW != initavgW:
            fctrA = curravgW / initavgW
            dochange = False
            if fctrA < MinFctr:
              fctrB = (MinFctr * initavgW) / curravgW # factor to restore weights to boundary
              dochange = True
            elif fctrA > MaxFctr:
              fctrB = (MaxFctr * initavgW) / curravgW # factor to restore weights to boundary
              dochange = True
            # print('initW:',initW,'currW:',currW,'fctr:',fctr)
            if dochange:
              dfctr[k] = fctrB
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
      fid3.write('\t%d' % sim.followTargetSign[i])
      fid3.write('\n')

######################################################################################

# adjusted from https://github.com/NathanKlineInstitute/netpyne-STDP/blob/master/neurosim/sim.py
def getSpikesWithInterval(trange=None, neuronal_pop=None):
  if len(neuronal_pop) < 1:
    return 0.0
  spkts = sim.simData['spkt']
  spkids = sim.simData['spkid']
  dminID = sim.simData['dminID']
  dmaxID = sim.simData['dmaxID']
  pop_spikes = {p:0 for p in neuronal_pop}
  if len(spkts) > 0:
    len_skts = len(spkids)
    for idx in range(len_skts):
      i = len_skts - 1 - idx
      if trange[0] <= spkts[i] <= trange[1]:
        for p in neuronal_pop:
          if spkids[i] >= dminID[p] and spkids[i] <= dmaxID[p]:
            pop_spikes[p] += 1
            break
      if trange[0] > spkts[i]: break
  return pop_spikes

""" old version
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
"""

NBsteps = 0 # this is a counter for recording the plastic weights
epCount = []
proposed_actions = [] 
total_hits = [] #numbertimes ball is hit by racket as ball changes its direction and player doesn't lose a score (assign 1). if player loses
dSTDPmech = {} # dictionary of list of STDP mechanisms

def InitializeNoiseRates ():
  # initialize the noise firing rates for the primary visual neuron populations (location V1 and direction sensitive)
  # based on image contents
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    #np.random.seed(1234)
    for pop in sim.lnoisety:
      if pop in sim.net.pops:
        for cell in sim.net.cells:
          if cell.gid in sim.net.pops[pop].cellGids:
            cell.hPointp.interval = 10
            cell.hPointp.start = 0 # np.random.uniform(0,1200) 
  else:
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

def InitializeInputRates ():
  # initialize the source firing rates for the primary visual neuron populations (location V1 and direction sensitive)
  # based on image contents
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7':
    np.random.seed(1234)
    for pop in sim.lstimty:
      if pop in sim.net.pops:
        for cell in sim.net.cells:
          if cell.gid in sim.net.pops[pop].cellGids:
            cell.hPointp.interval = 1e12
            cell.hPointp.start = 0 # np.random.uniform(0,1200) 
  else:
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
  # update the source firing rates for the primary visual neuron populations (location V1 and direction sensitive)
  # based on image contents
  # this py_alltoall seems to work, but is apparently not as fast as py_broadcast (which has problems - wrong-sized array sometimes!)
  root = 0
  nhost = sim.pc.nhost()
  src = [sim.AIGame.dFiringRates]*nhost if sim.rank == root else [None]*nhost
  dFiringRates = sim.pc.py_alltoall(src)[0]
  if sim.rank == 0: dFiringRates = sim.AIGame.dFiringRates
  # if sim.rank==0: print(dFiringRates['EV1'])  
  # update input firing rates for stimuli to ER,EV1 and direction sensitive cells
  if ECellModel == 'IntFire4' or ECellModel == 'INTF7': # different rules/code when dealing with artificial cells
    lsz = len('stimMod') # this is a prefix
    for pop in sim.lstimty: # go through NetStim populations
      if pop in sim.net.pops: # make sure the population exists
        lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of NetStim cells
        offset = sim.simData['dminID'][pop]
        #print(pop,pop[lsz:],offset)
        for cell in lCell:
          if dFiringRates[pop[lsz:]][int(cell.gid-offset)]==0:
            cell.hPointp.interval = 1e12
          else:
            cell.hPointp.interval = 1000.0/dFiringRates[pop[lsz:]][int(cell.gid-offset)] #40 
  else:
    if dnumc['ER']>0:
      lratepop = ['ER', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']
    else:
      lratepop = ['EV1', 'EV1DE', 'EV1DNE', 'EV1DN', 'EV1DNW', 'EV1DW', 'EV1DSW', 'EV1DS', 'EV1DSE']    
    for pop in lratepop:
      if dnumc[pop] <= 0: continue
      lCell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells
      offset = sim.simData['dminID'][pop]
      #if dconf['verbose'] > 1: print(sim.rank,'updating len(',pop,len(lCell),'source firing rates. len(dFiringRates)=',len(dFiringRates[pop]))
      for cell in lCell:  
        for stim in cell.stims:
          if stim['source'] == 'stimMod':
            if dFiringRates[pop][int(cell.gid-offset)]==0:
              stim['hObj'].interval = 1e12
            else:
              stim['hObj'].interval = 1000.0/dFiringRates[pop][int(cell.gid-offset)] #40 #fchoices[rind] #10 #

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
          cCellW+=conn['hObj'].weight[PlastWeightIndex]
      sim.dTargetW[cell.gid] = cCellW

def getFiringRateWithInterval (trange, neuronal_pop):
  if len(neuronal_pop) < 1: return 0.0
  spkts = np.array(sim.simData['spkt'])
  spkids = np.array(sim.simData['spkid'])
  ctspkids = spkids[(spkts>=trange[0])&(spkts <= trange[1])]
  pop_firingrates = {cellid:0 for cellid in neuronal_pop}
  if len(spkts)>0:
    for cellid in neuronal_pop:
      pop_firingrates[cellid] = 1000.0*len(np.where(ctspkids==cellid)[0])/(trange[1]-trange[0])
  return pop_firingrates

def getFiringRateWithIntervalAllNeurons (sim, trange, lpop):
  lgid = []
  for pop in lpop:
    for gid in sim.net.pops[pop].cellGids:
      lgid.append(gid)
  sim.dFR = getFiringRateWithInterval(trange, lgid)

def adjustTargetWBasedOnFiringRates (sim):
  dshift = dconf['net']['homPlast']['dshift'] # shift in weights to push within min,max firing rate bounds
  dscale = dconf['net']['homPlast']['dscale'] # shift in weights to push within min,max firing rate bounds  
  for gid,cTargetW in sim.dTargetW.items():
    cTargetFRMin, cTargetFRMax = sim.dMINTargetFR[gid], sim.dMAXTargetFR[gid]
    if gid not in sim.dFR: continue
    cFR = sim.dFR[gid] # current cell firing rate
    if cFR>cTargetFRMax: # above max threshold firing rate
      #print('Target W DOWN',sim.rank,gid,'rate=',cFR,sim.dTargetW[gid],sim.dTargetW[gid] * (1.0 - dscale))
      if dshift != 0.0:
        sim.dTargetW[gid] -= dshift
      elif dscale != 0.0:
        sim.dTargetW[gid] *= (1.0 - dscale)
      else:
        sim.dTargetW[gid] *= cTargetFRMax / cFR
    elif cFR<cTargetFRMin: # below min threshold firing rate
      #print('Target W UP',sim.rank,gid,'rate=',cFR,sim.dTargetW[gid],sim.dTargetW[gid] * (1.0 + dscale))
      if dshift != 0.0:
        sim.dTargetW[gid] += dshift
      elif dscale != 0.0:
        sim.dTargetW[gid] *= (1.0 + dscale)
      else:
        sim.dTargetW[gid] *= cTargetFRMin / cFR 
  return sim.dTargetW

def adjustWeightsBasedOnFiringRates (sim,lpop,synType='AMPA'):
  # normalize the STDP/RL weights during the simulation - called in trainAgent
  countScaleUps = 0
  countScaleDowns = 0
  for pop in lpop:
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[pop].cellGids] # this is the set of cells in a pop
    for cell in lcell:
      cFR = sim.dFR[cell.gid]
      if cFR >= sim.dMINTargetFR[cell.gid] and cFR <= sim.dMAXTargetFR[cell.gid]: continue # only change weights if firing rate outside bounds
      targetW = sim.dTargetW[cell.gid]
      cCellW = 0
      for conn in cell.conns:
        if conn.synMech==synType and 'hSTDP' in conn: # to make sure that only specific type of synapses are counted towards the sum.
          cCellW+=conn['hObj'].weight[PlastWeightIndex]
      if cCellW>0: # if no weight associated with the specific type of synapses, no need to asdjust weights.
        sfctr = targetW/cCellW
        if sfctr>1:
          countScaleUps += 1
          #print('W UP',sim.rank,cell.gid,sfctr)
        elif sfctr<1:
          countScaleDowns += 1
          #print('W DOWN',sim.rank,cell.gid,sfctr)
        if sfctr != 1.0:
          for conn in cell.conns:
            if conn.synMech==synType and 'hSTDP' in conn:
              conn['hObj'].weight[PlastWeightIndex] *= sfctr
  print(sim.rank,'adjust W: UP=', countScaleUps, ', DOWN=', countScaleDowns)

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
  if dconf['net']['useBinaryImage']:
    #InputImages = np.where(InputImages>0,1,0)
    """
    with open('data/'+dconf['sim']['name']+'InputImages.txt', 'w') as outfile:
      outfile.write('# Array shape: {0}\n'.format(InputImages.shape))
      for Input_Image in InputImages:
        np.savetxt(outfile, Input_Image, fmt='%d', delimiter=' ')
        outfile.write('# New slice\n')
    """
    np.save('data/'+dconf['sim']['name']+'InputImages',InputImages)
  else:
    with open('data/'+dconf['sim']['name']+'InputImages.txt', 'w') as outfile:
      outfile.write('# Array shape: {0}\n'.format(InputImages.shape))
      for Input_Image in InputImages:
        np.savetxt(outfile, Input_Image, fmt='%-7.2f', delimiter=' ')
        outfile.write('# New slice\n')  
  
def finishSim ():        
  if sim.rank==0 and fid4 is not None: fid4.close()
  if ECellModel == 'INTF7' or ICellModel == 'INTF7': intf7.insertSpikes(sim, simConfig.recordStep)  
  sim.gatherData() # gather data from different nodes
  sim.saveData() # save data to disk    
  if sim.saveWeights: saveSynWeights()
  # only rank 0 should save. otherwise all the other nodes could over-write the output or quit first; rank 0 plots  
  if sim.rank == 0: 
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
  
def trainAgent (t):
  """ training interface between simulation and game environment
  """
  global NBsteps, epCount, proposed_actions, total_hits, fid4, tstepPerAction
  critic = 0
  vec = h.Vector()
  if t<(tstepPerAction*dconf['actionsPerPlay']): # for the first time interval use randomly selected actions
    actions =[]
    for _ in range(int(dconf['actionsPerPlay'])):
      action = dconf['movecodes'][random.randint(0,len(dconf['movecodes'])-1)]
      actions.append(action)
  else: #the actions should be based on the activity of motor cortex (EMUP, EMDOWN)
    F_UPs = []
    F_DOWNs = []
    for ts in range(int(dconf['actionsPerPlay'])):
      ts_beg = t-tstepPerAction*(dconf['actionsPerPlay']-ts-1) 
      ts_end = t-tstepPerAction*(dconf['actionsPerPlay']-ts)
      dfreq = getSpikesWithInterval([ts_end,ts_beg], ['EMUP', 'EMDOWN'])
      F_UPs.append(dfreq['EMUP'])
      F_DOWNs.append(dfreq['EMDOWN'])
      #F_UPs.append(getSpikesWithInterval([ts_end,ts_beg], sim.net.pops['EMUP'].cellGids))
      #F_DOWNs.append(getSpikesWithInterval([ts_end,ts_beg], sim.net.pops['EMDOWN'].cellGids))
    sim.pc.allreduce(vec.from_python(F_UPs),1) #sum
    F_UPs = vec.to_python()
    sim.pc.allreduce(vec.from_python(F_DOWNs),1) #sum
    F_DOWNs = vec.to_python()
    if sim.rank==0:
      if fid4 is None: fid4 = open(sim.MotorOutputsfilename,'w')
      print('t=',round(t,2),' U,D spikes:', F_UPs, F_DOWNs)
      fid4.write('%0.1f' % t)
      for ts in range(int(dconf['actionsPerPlay'])): fid4.write('\t%0.1f' % F_UPs[ts])
      for ts in range(int(dconf['actionsPerPlay'])): fid4.write('\t%0.1f' % F_DOWNs[ts])
      fid4.write('\n')
      actions = []
      if dconf['randmove']:
        lmoves = list(dconf['moves'].values())
        for ts in range(int(dconf['actionsPerPlay'])): actions.append(lmoves[np.random.randint(0,len(lmoves))])
      elif dconf['stochmove']:
        if random.uniform(0,1) <= dconf['stochmove']:
          lmoves = [dconf['moves']['UP'], dconf['moves']['DOWN'], dconf['moves']['NOMOVE']]
          actions.append(lmoves[np.random.randint(0,len(lmoves))])
          print('stochastic move = ', actions[-1])
        else:
          if F_UPs[ts]>F_DOWNs[ts]: # UP WINS
            actions.append(dconf['moves']['UP'])
          elif F_DOWNs[ts]>F_UPs[ts]: # DOWN WINS
            actions.append(dconf['moves']['DOWN'])
          else:
            actions.append(dconf['moves']['NOMOVE'])
        sim.lastMove = actions[-1]
      else:
        for ts in range(int(dconf['actionsPerPlay'])):
          if F_UPs[ts]>F_DOWNs[ts]: # UP WINS
            actions.append(dconf['moves']['UP'])
          elif F_DOWNs[ts]>F_UPs[ts]: # DOWN WINS
            actions.append(dconf['moves']['DOWN'])
          elif dconf['0rand'] and F_DOWNs[ts]==0 and F_UPs[ts]==0: # random move when 0 rate for both pops?
            lmoves = list(dconf['moves'].values())
            actions.append(lmoves[np.random.randint(0,len(lmoves))])
          else:
            actions.append(dconf['moves']['NOMOVE']) # No move
  if sim.rank == 0:
    rewards, epCount, proposed_actions, total_hits, FollowTargetSign = sim.AIGame.playGame(actions, epCount, t)
    dback = {4:'UP',3:'DOWN',1:'STAY',-1:'NOP'}
    print('t=',round(t,2),'proposed,model action:', [dback[x] for x in proposed_actions],[dback[x] for x in actions])
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
    if dconf['useFollowMoveOutput']:
      for caction, cproposed_action in zip(actions, proposed_actions):
        if cproposed_action == -1: # invalid action since e.g. ball not visible
          continue
        elif FollowTargetSign > 0: # model moved racket towards predicted y intercept - gets a reward
          critic_for_following_ball += dconf['rewardcodes']['followTarget'] #follow the ball
        elif FollowTargetSign < 0: # model moved racket away from predicted y intercept - gets a punishment
          critic_for_following_ball += dconf['rewardcodes']['avoidTarget'] # didn't follow the ball        
    else:
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
  else: # other workers
    critic = sim.pc.py_broadcast(None, 0) # receive critic value from master node
    UPactions = sim.pc.py_broadcast(None, 0)
    DOWNactions = sim.pc.py_broadcast(None, 0)
    if dconf['verbose']>1:
      print('UPactions: ', UPactions,'DOWNactions: ', DOWNactions)      
  if critic != 0: # if critic signal indicates punishment (-1) or reward (+1)
    if sim.rank==0: print('t=',round(t,2),'RLcritic:',critic)
    if dconf['sim']['targettedRL']:
      if UPactions==DOWNactions and \
         sum(F_UPs)>0 and sum(F_DOWNs)>0: # same number of actions/spikes -> stay; only apply critic when > 0 spikes
        if dconf['verbose']: print('APPLY RL to both EMUP and EMDOWN')
        if dconf['sim']['targettedRL']>=4:
          for STDPmech in dSTDPmech['EM']: STDPmech.reward_punish(float(critic)) # EM populations get reward/punishment on a tie            
          for STDPmech in dSTDPmech['nonEM']: STDPmech.reward_punish(float(dconf['sim']['targettedRLDscntFctr']*critic)) # but non-EM get less than EM
        else: # usual targetted RL (==1 or ==3)
          for STDPmech in dSTDPmech['all']: STDPmech.reward_punish(float(critic))          
      elif UPactions>DOWNactions: # UP WINS vs DOWN
        if dconf['verbose']: print('APPLY RL to EMUP')
        for STDPmech in dSTDPmech['EMUP']: STDPmech.reward_punish(float(critic))
        if dconf['sim']['targettedRL']>=3 and sum(F_DOWNs)>0: # opposite to pop that did not contribute
          if dconf['verbose']: print('APPLY -RL to EMDOWN')
          for STDPmech in dSTDPmech['EMDOWN']: STDPmech.reward_punish(float(-dconf['sim']['targettedRLOppFctr']*critic))
        if dconf['sim']['targettedRL']>=4: # apply to non-EM with a discount factor
          for STDPmech in dSTDPmech['nonEM']: STDPmech.reward_punish(float(dconf['sim']['targettedRLDscntFctr']*critic))
      elif DOWNactions>UPactions: # DOWN WINS vs UP
        if dconf['verbose']: print('APPLY RL to EMDOWN')
        for STDPmech in dSTDPmech['EMDOWN']: STDPmech.reward_punish(float(critic))
        if dconf['sim']['targettedRL']>=3 and sum(F_UPs)>0: # opposite to pop that did not contribute
          if dconf['verbose']: print('APPLY -RL to EMUP')            
          for STDPmech in dSTDPmech['EMUP']: STDPmech.reward_punish(float(-dconf['sim']['targettedRLOppFctr']*critic))
        if dconf['sim']['targettedRL']>=4: # apply to non-EM with a discount factor
          for STDPmech in dSTDPmech['nonEM']: STDPmech.reward_punish(float(dconf['sim']['targettedRLDscntFctr']*critic))  
    else: # this is non-targetted RL
      if dconf['verbose']: print('APPLY RL to both EMUP and EMDOWN')
      for STDPmech in dSTDPmech['all']: STDPmech.reward_punish(critic)
      for STDPmech in dSTDPmech['NOISE']: STDPmech.reward_punish(-critic) # noise sources get opposite sign RL
    if dconf['sim']['ResetEligAfterCritic']: # reset eligibility after applying reward/punishment
      for STDPmech in dSTDPmech['all']: STDPmech.reset_eligibility()
      for STDPmech in dSTDPmech['NOISE']: STDPmech.reset_eligibility()
  if sim.rank==0:
    # print('t=',round(t,2),' game rewards:', rewards) # only rank 0 has access to rewards      
    for action in actions: sim.allActions.append(action)
    for pactions in proposed_actions: sim.allProposedActions.append(pactions) #also record proposed actions
    for reward in rewards: sim.allRewards.append(reward)
    for hits in total_hits: sim.allHits.append(hits) # hit or no hit
    sim.followTargetSign.append(FollowTargetSign) 
    tvec_actions = []
    for ts in range(len(actions)): tvec_actions.append(t-tstepPerAction*(len(actions)-ts-1))
    for ltpnt in tvec_actions: sim.allTimes.append(ltpnt)
  updateInputRates() # update firing rate of inputs to R population (based on image content)                
  NBsteps += 1
  if NBsteps % recordWeightStepSize == 0:
    if dconf['verbose'] > 0 and sim.rank==0:
      print('Weights Recording Time:', t, 'NBsteps:',NBsteps,'recordWeightStepSize:',recordWeightStepSize)
    recordAdjustableWeights(sim, t, lrecpop) 
  if NBsteps % normalizeWeightStepSize == 0:
    if dconf['verbose'] > 0 and sim.rank==0:
      print('Weight Normalize Time:', t, 'NBsteps:',NBsteps,'normalizeWeightStepSize:',normalizeWeightStepSize)
    sim.pc.barrier()
    normalizeAdjustableWeights(sim, t, lrecpop)
    sim.pc.barrier()    
  if dconf['net']['homPlast']['On']:
    if NBsteps % dconf['net']['homPlast']['hsIntervalSteps'] == 0:
      if sim.rank==0: print('adjustTargetWBasedOnFiringRates')
      hsInterval = tstepPerAction*dconf['actionsPerPlay']*dconf['net']['homPlast']['hsIntervalSteps']
      getFiringRateWithIntervalAllNeurons(sim, [t-hsInterval,t], sim.dHPlastPops) # call this function at hsInterval
      adjustTargetWBasedOnFiringRates(sim) # call this function at hsInterval
    if NBsteps % dconf['net']['homPlast']['updateIntervalSteps'] == 0:
      if sim.rank==0: print('adjustWeightsBasedOnFiringRates')
      adjustWeightsBasedOnFiringRates(sim,sim.dHPlastPops,synType=dconf['net']['homPlast']['synType'])
      sim.pc.barrier()
  if dconf['sim']['QuitAfterMiss'] and critic < 0.0: finishSim()
        
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

def setrecspikes ():
  if dconf['sim']['recordStim']:
    sim.cfg.recordCellsSpikes = [-1] # record from all spikes
  else:
    # make sure to record only from the neurons, not the stimuli - which requires a lot of storage/memory
    sim.cfg.recordCellsSpikes = []
    for pop in sim.net.pops.keys():
      if pop.count('stim') > 0 or pop.count('Noise') > 0: continue
      for gid in sim.net.pops[pop].cellGids: sim.cfg.recordCellsSpikes.append(gid)

setrecspikes()
sim.setupRecording()                  # setup variables to record for each cell (spikes, V traces, etc)

def setdminmaxID (sim, lpop):
  # setup min,max ID for each population in lpop
  alltags = sim._gatherAllCellTags() #gather cell tags; see https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/sim/gather.py
  dGIDs = {pop:[] for pop in lpop}
  for tinds in range(len(alltags)):
    if alltags[tinds]['pop'] in lpop:
      dGIDs[alltags[tinds]['pop']].append(tinds)
  sim.simData['dminID'] = {pop:np.amin(dGIDs[pop]) for pop in lpop if len(dGIDs[pop])>0}
  sim.simData['dmaxID'] = {pop:np.amax(dGIDs[pop]) for pop in lpop if len(dGIDs[pop])>0} 

setdminmaxID(sim, allpops) # this needs to be called before getALLSTDPObjects (since uses dminID,dmaxID for EN populations when they're present)

def getAllSTDPObjects (sim):
  # get all the STDP objects from the simulation's cells
  Mpops = ['EMUP', 'EMDOWN']  
  dSTDPmech = {'all':[]} # dictionary of STDP objects keyed by type (all, for EMUP, EMDOWN populations) -- excludes NOISE RL (see below)
  for pop in Mpops: dSTDPmech[pop] = []
  if dconf['sim']['targettedRL']:
    if dconf['sim']['targettedRL']>=4:
      dSTDPmech['nonEM'] = [] # not post-synapse of an EM neuron (only used for targetted RL when RL plasticity at non-EM neurons)
      dSTDPmech['EM'] = [] # post-synapse of an EM neuron (EMDOWN or EMUP, etc.)
  dSTDPmech['NOISE'] = [] # for noise RL (presynaptic source is noisy neuron)
  # print('sim.rank=',sim.rank,'len(sim.net.pops[EN].cellGids)=',len(sim.net.pops['EN'].cellGids),dnumc['EN'])
  if 'EN' in sim.net.pops:
    for cell in sim.net.cells:
      for conn in cell.conns:
        STDPmech = conn.get('hSTDP')  # check if the connection has a NEURON STDP mechanism object
        if STDPmech:
          preNoise = False
          cpreID = conn.preGid  #find preID
          if type(cpreID) == int:
            if cpreID >= sim.simData['dminID']['EN'] and cpreID <= sim.simData['dmaxID']['EN']:
              preNoise = True
              # print('found preNoise')
          if preNoise:
            dSTDPmech['NOISE'].append(STDPmech)
          else:
            dSTDPmech['all'].append(STDPmech)
            isEM = False
            for pop in Mpops:
              if cell.gid in sim.net.pops[pop].cellGids:
                dSTDPmech[pop].append(STDPmech)
                isEM = True
                if dconf['sim']['targettedRL']>=4: dSTDPmech['EM'].append(STDPmech) # any EM
            if dconf['sim']['targettedRL']>=4:
              if not isEM: dSTDPmech['nonEM'].append(STDPmech)    
  else:
    for cell in sim.net.cells:
      for conn in cell.conns:
        STDPmech = conn.get('hSTDP')  # check if the connection has a NEURON STDP mechanism object
        if STDPmech:
          dSTDPmech['all'].append(STDPmech)
          isEM = False
          for pop in Mpops:
            if cell.gid in sim.net.pops[pop].cellGids:
              dSTDPmech[pop].append(STDPmech)
              isEM = True
              if dconf['sim']['targettedRL']>=4: dSTDPmech['EM'].append(STDPmech) # any EM
          if dconf['sim']['targettedRL']>=4:
            if not isEM: dSTDPmech['nonEM'].append(STDPmech)
  return dSTDPmech

dSTDPmech = getAllSTDPObjects(sim) # get all the STDP objects up-front

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
        conn['hObj'].weight[PlastWeightIndex] = cW
        #hSTDP = conn.get('hSTDP')
        #hSTDP.cumreward = cConnW.at[idx,'cumreward']
        if dconf['verbose'] > 1: print('weight updated:', cW)
        
#if specified 'ResumeSim' = 1, load the connection data from 'ResumeSimFromFile' and assign weights to STDP synapses  
if dconf['simtype']['ResumeSim']:
  try:
    from simdat import readweightsfile2pdf
    A = readweightsfile2pdf(dconf['simtype']['ResumeSimFromFile'])
    updateSTDPWeights(sim, A[A.time == max(A.time)]) # take the latest weights saved    
    sim.pc.barrier() # wait for other nodes
    if sim.rank==0: print('Updated STDP weights')    
    if 'normalizeWeightsAtStart' in dconf['sim']:
      if dconf['sim']['normalizeWeightsAtStart']:
        normalizeAdjustableWeights(sim, 0, lrecpop)
        print(sim.rank,'normalized adjustable weights at start')
        sim.pc.barrier() # wait for other nodes
  except:
    print('Could not restore STDP weights from file.')

if dconf['net']['homPlast']['On']:
  # call this once before the simulation
  initTargetFR(sim,dconf['net']['homPlast']['mintargetFR'],dconf['net']['homPlast']['maxtargetFR'])
  # call this once before running the simulation.    
  initTargetW(sim,list(dconf['net']['homPlast']['mintargetFR'].keys()),synType=dconf['net']['homPlast']['synType']) 

tPerPlay = tstepPerAction*dconf['actionsPerPlay']
InitializeInputRates()
#InitializeNoiseRates()
dsumWInit = getSumAdjustableWeights(sim) # get sum of adjustable weights at start of sim

sim.runSimWithIntervalFunc(tPerPlay,trainAgent) # has periodic callback to adjust STDP weights based on RL signal
finishSim() # gather data, save, plot (optional), quit
