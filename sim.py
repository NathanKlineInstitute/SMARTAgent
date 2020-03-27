from netpyne import specs, sim
from neuron import h
import numpy as np
import random
from conf import dconf # configuration dictionary
import pandas as pd
import pickle
from collections import OrderedDict

random.seed(1234) # this will not work properly across runs with different number of nodes

sim.allTimes = []
sim.allRewards = [] # list to store all rewards
sim.allActions = [] # list to store all actions
sim.allProposedActions = [] # list to store all proposed actions
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
recordWeightStepSize = dconf['sim']['recordWeightStepSize']
#recordWeightDT = 1000 # interval for recording synaptic weights (change later)
recordWeightDCells = 1 # to record weights for sub samples of neurons

global fid4

fid4 = open(sim.MotorOutputsfilename,'w')

scale = dconf['net']['scale']
ETypes = ['ER','EV1','EV1D0','EV1D90','EV1D180','EV1D270','EV4','EIT', 'EML', 'EMR']
ITypes = ['IR','IV1','IV4','IIT','IM']
allpops = ['ER','IR','EV1','EV1D0','EV1D90','EV1D180','EV1D270','IV1','EV4','IV4','EIT','IIT','EML','EMR','IM']
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

#For AMPA-faster synapses
STDPparamsRL1 = {'hebbwt': 0.0000, 'antiwt':-0.0000, 'wbase': 0.0005, 'wmax': 1, 'RLon': 1 , 'RLhebbwt': 0.001 , 'RLantiwt': -0.000,
                'tauhebb': 10, 'RLlenhebb': 50 ,'RLlenanti': 50, 'RLwindhebb': 50, 'useRLexp': 1, 'softthresh': 0, 'verbose':0}
#for NMDA (slower) synapses
STDPparamsRL2 = {'hebbwt': 0.0000, 'antiwt':-0.0000, 'wbase': 0.0005, 'wmax': 1, 'RLon': 1 , 'RLhebbwt': 0.001 , 'RLantiwt': -0.000,
                'tauhebb': 10, 'RLlenhebb': 800 ,'RLlenanti': 100, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 0, 'verbose':0}

# these are the image-based inputs provided to the R (retinal) cells
netParams.stimSourceParams['stimMod'] = {'type': 'NetStim', 'rate': 'variable', 'noise': 0}
netParams.stimTargetParams['stimMod->all'] = {'source': 'stimMod',
        'conds': {'pop': ['ER','EV1D0','EV1D90','EV1D180','EV1D270']},
        'convergence': 1,
        'weight': 1,
        'delay': 1,
        'synMech': 'AMPA'}

#background input to inhibitory neurons to increase their firing rate

# Stimulation parameters

netParams.stimSourceParams['ebkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['ebkg->all'] = {'source': 'ebkg', 'conds': {'cellType': ['EV1','EV4','EIT']}, 'weight': 0.0, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'}

netParams.stimSourceParams['MLbkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['MLbkg->all'] = {'source': 'MLbkg', 'conds': {'cellType': ['EML']}, 'weight': 0.0, 'delay': 1, 'synMech': 'AMPA'}

netParams.stimSourceParams['MRbkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 1.0}
netParams.stimTargetParams['MRbkg->all'] = {'source': 'MRbkg', 'conds': {'cellType': ['EMR']}, 'weight': 0.0, 'delay': 1, 'synMech': 'AMPA'}

netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 20, 'noise': 1.0}
netParams.stimTargetParams['bkg->all'] = {'source': 'bkg', 'conds': {'cellType': ['IR','IV1','IV4','IIT']}, 'weight': 0.0, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'}

######################################################################################
def connectLayerswithOverlap(NBpreN, NBpostN, overlap_xdir):
    #NBpreN = 6400 	#number of presynaptic neurons
    NBpreN_x = int(np.sqrt(NBpreN))
    NBpreN_y = int(np.sqrt(NBpreN))
    #NBpostN = 6400	#number of postsynaptic neurons
    NBpostN_x = int(np.sqrt(NBpostN))
    NBpostN_y = int(np.sqrt(NBpostN))
    convergence_factor = NBpreN/NBpostN
    convergence_factor_x = np.ceil(np.sqrt(convergence_factor))
    convergence_factor_y = np.ceil(np.sqrt(convergence_factor))
    #overlap_xdir = 5	#number of rows in a window for overlapping connectivity
    #overlap_ydir = 5	#number of columns in a window for overlapping connectivity
    overlap_ydir = overlap_xdir
    preNIndices = np.zeros((NBpreN_x,NBpreN_y))
    postNIndices = np.zeros((NBpostN_x,NBpostN_y))		#list created for indices from linear (1-6400) to square indexing (1-80,81-160,....) 
    blist = []
    for i in range(NBpreN_x):
        for j in range(NBpreN_y):
            preNIndices[i,j]=j+(NBpreN_y*i)
    for i in range(NBpostN_x):
        for j in range(NBpostN_y):
            postNIndices[i,j]=j+(NBpostN_y*i)
    for i in range(NBpostN_x):				#boundary conditions are implemented here
        for j in range(NBpostN_y):
            postN = int(postNIndices[i,j])
            if convergence_factor_x>1:
                preN = preNIndices[int(i*convergence_factor_y),int(j*convergence_factor_x)]
                #preN = int(convergence_factor_x*convergence_factor_y*NBpostN_y*i) + int(convergence_factor_y*j)
            else:
                preN = int(postN)
            preN_ind = np.where(preNIndices==preN)
            x0 = preN_ind[0][0] - int(overlap_xdir/2)
            if x0<0:
                x0 = 0
            y0 = preN_ind[1][0] - int(overlap_ydir/2)
            if y0<0:
                y0 = 0
            xlast = preN_ind[0][0] + int(overlap_xdir/2)
            if xlast>NBpreN_x-1:
                xlast = NBpreN_x-1
            ylast = preN_ind[1][0] + int(overlap_ydir/2)
            if ylast>NBpreN_y-1:
                ylast = NBpreN_y-1
            xinds = [x0]
            for _ in range(xlast-x0):
                xinds.append(xinds[-1]+1)
            yinds = [y0]
            for _ in range(ylast-y0):
                yinds.append(yinds[-1]+1)
            for xi in range(len(xinds)):
                for yi in range(len(yinds)):
                    preN = int(preNIndices[xinds[xi],yinds[yi]])
                    blist.append([preN,postN]) 			#list of [presynaptic_neuron, postsynaptic_neuron] 
    return blist

def connectLayerswithOverlapDiv(NBpreN, NBpostN, overlap_xdir):
    NBpreN_x = int(np.sqrt(NBpreN))
    NBpreN_y = int(np.sqrt(NBpreN))
    NBpostN_x = int(np.sqrt(NBpostN))
    NBpostN_y = int(np.sqrt(NBpostN))
    divergence_factor = NBpostN/NBpreN
    divergence_factor_x = np.ceil(np.sqrt(divergence_factor))
    divergence_factor_y = np.ceil(np.sqrt(divergence_factor))
    overlap_ydir = overlap_xdir
    preNIndices = np.zeros((NBpreN_x,NBpreN_y))
    postNIndices = np.zeros((NBpostN_x,NBpostN_y))		#list created for indices from linear (1-6400) to square indexing (1-80,81-160,....) 
    blist = []
    for i in range(NBpreN_x):
        for j in range(NBpreN_y):
            preNIndices[i,j]=j+(NBpreN_y*i)
    for i in range(NBpostN_x):
        for j in range(NBpostN_y):
            postNIndices[i,j]=j+(NBpostN_y*i)
    for i in range(NBpreN_x):				#boundary conditions are implemented here
        for j in range(NBpreN_y):
            preN = int(preNIndices[i,j])
            if divergence_factor_x>1:
                postN = postNIndices[int(i*divergence_factor_y),int(j*divergence_factor_x)]
            else:
                postN = int(preN)
            postN_ind = np.where(postNIndices==postN)
            x0 = postN_ind[0][0] - int(overlap_xdir/2)
            if x0<0:
                x0 = 0
            y0 = postN_ind[1][0] - int(overlap_ydir/2)
            if y0<0:
                y0 = 0
            xlast = postN_ind[0][0] + int(overlap_xdir/2)
            if xlast>NBpostN_x-1:
                xlast = NBpostN_x-1
            ylast = postN_ind[1][0] + int(overlap_ydir/2)
            if ylast>NBpostN_y-1:
                ylast = NBpostN_y-1
            xinds = [x0]
            for _ in range(xlast-x0):
                xinds.append(xinds[-1]+1)
            yinds = [y0]
            for _ in range(ylast-y0):
                yinds.append(yinds[-1]+1)
            for xi in range(len(xinds)):
                for yi in range(len(yinds)):
                    postN = int(postNIndices[xinds[xi],yinds[yi]])
                    blist.append([preN,postN]) 			#list of [presynaptic_neuron, postsynaptic_neuron] 
    return blist

#####################################################################################
#Feedforward excitation
#E to E - Feedforward connections
blistERtoEV1 = connectLayerswithOverlap(NBpreN = dnumc['ER'], NBpostN = dnumc['EV1'], overlap_xdir = 3)
blistEV1toEV4 = connectLayerswithOverlap(NBpreN = dnumc['EV1'], NBpostN = dnumc['EV4'], overlap_xdir = 3)
blistEV4toEIT = connectLayerswithOverlap(NBpreN = dnumc['EV4'], NBpostN = dnumc['EIT'], overlap_xdir = 3) #was 15
#blistITtoMI = connectLayerswithOverlap(NBpreN = NB_ITneurons, NBpostN = NB_MIneurons, overlap_xdir = 3) #Not sure if this is a good strategy instead of all to all
#blistMItoMO = connectLayerswithOverlap(NBpreN = NB_MIneurons, NBpostN = NB_MOneurons, overlap_xdir = 3) #was 19
#blistMItoMO: Feedforward for MI to MO is all to all and can be specified in the connection statement iteself

#E to I - Feedforward connections
blistERtoIV1 = connectLayerswithOverlap(NBpreN = dnumc['ER'], NBpostN = dnumc['IV1'], overlap_xdir = 3)
blistEV1toIV4 = connectLayerswithOverlap(NBpreN = dnumc['EV1'], NBpostN = dnumc['IV4'], overlap_xdir = 3) 
blistEV4toIIT = connectLayerswithOverlap(NBpreN = dnumc['EV4'], NBpostN = dnumc['IIT'], overlap_xdir = 3) 

#E to I - WithinLayer connections
blistERtoIR = connectLayerswithOverlap(NBpreN = dnumc['ER'], NBpostN = dnumc['IR'], overlap_xdir = 3)
blistEV1toIV1 = connectLayerswithOverlap(NBpreN = dnumc['EV1'], NBpostN = dnumc['IV1'], overlap_xdir = 3)
blistEV4toIV4 = connectLayerswithOverlap(NBpreN = dnumc['EV4'], NBpostN = dnumc['IV4'], overlap_xdir = 3)
blistEITtoIIT = connectLayerswithOverlap(NBpreN = dnumc['EIT'], NBpostN = dnumc['IIT'], overlap_xdir = 3)

#I to E - WithinLayer Inhibition
blistIRtoER = connectLayerswithOverlapDiv(NBpreN = dnumc['IR'], NBpostN = dnumc['ER'], overlap_xdir = 5)
blistIV1toEV1 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV1'], NBpostN = dnumc['EV1'], overlap_xdir = 5)
blistIV4toEV4 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV4'], NBpostN = dnumc['EV4'], overlap_xdir = 5)
blistIITtoEIT = connectLayerswithOverlapDiv(NBpreN = dnumc['IIT'], NBpostN = dnumc['EIT'], overlap_xdir = 5)

#Feedbackward excitation
#E to E  
blistEV1toER = connectLayerswithOverlapDiv(NBpreN = dnumc['EV1'], NBpostN = dnumc['ER'], overlap_xdir = 3)
blistEV4toEV1 = connectLayerswithOverlapDiv(NBpreN = dnumc['EV4'], NBpostN = dnumc['EV1'], overlap_xdir = 3)
blistEITtoEV4 = connectLayerswithOverlapDiv(NBpreN = dnumc['EIT'], NBpostN = dnumc['EV4'], overlap_xdir = 3)

#Feedforward inhibition
#I to I
blistIV1toIV4 = connectLayerswithOverlap(NBpreN = dnumc['IV1'], NBpostN = dnumc['IV4'], overlap_xdir = 5)
blistIV4toIIT = connectLayerswithOverlap(NBpreN = dnumc['IV4'], NBpostN = dnumc['IIT'], overlap_xdir = 5)

#Feedbackward inhibition
#I to E 
blistIV1toER = connectLayerswithOverlapDiv(NBpreN = dnumc['IV1'], NBpostN = dnumc['ER'], overlap_xdir = 5)
blistIV4toEV1 = connectLayerswithOverlapDiv(NBpreN = dnumc['IV4'], NBpostN = dnumc['EV1'], overlap_xdir = 5)
blistIITtoEV4 = connectLayerswithOverlapDiv(NBpreN = dnumc['IIT'], NBpostN = dnumc['EV4'], overlap_xdir = 5)

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
simConfig.savePickle = True            # Save params, network and sim output to pickle file
simConfig.saveMat = False
simConfig.saveFolder = 'data'
# simConfig.backupCfg = ['sim.json', 'backupcfg/'+dconf['sim']['name']+'sim.json']

#simConfig.analysis['plotRaster'] = True                         # Plot a raster
simConfig.analysis['plotTraces'] = {'include': [(pop, 0) for pop in allpops]}

#simConfig.analysis['plotRaster'] = {'timeRange': [500,1000],'popRates':'overlay','saveData':'data/RasterData.pkl','showFig':True}
simConfig.analysis['plotRaster'] = {'popRates':'overlay','saveData':'data/'+dconf['sim']['name']+'RasterData.pkl','showFig':dconf['sim']['doplot']}
#simConfig.analysis['plot2Dnet'] = True 
#simConfig.analysis['plotConn'] = True           # plot connectivity matrix

# synaptic weight gain (based on E, I types)
cfg = simConfig
cfg.EEGain = 0.75  # E to E scaling factor
cfg.EIGain = 1.0 # E to I scaling factor
cfg.IEGain = 10.0 # I to E scaling factor
cfg.IIGain = 10.0  # I to I scaling factor


#Local excitation
#E to E
netParams.connParams['ER->ER'] = {
        'preConds': {'pop': 'ER'},
        'postConds': {'pop': 'ER'},
        'probability': 0.02,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA', 'sec':'dend', 'loc':0.5}
netParams.connParams['EV1->EV1'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'EV1'},
        'probability': 0.02,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA', 'sec':'dend', 'loc':0.5}
netParams.connParams['EV4->EV4'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EV4'},
        'probability': 0.02,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA', 'sec':'dend', 'loc':0.5}
netParams.connParams['EIT->EIT'] = {
        'preConds': {'pop': 'EIT'},
        'postConds': {'pop': 'EIT'},
        'probability': 0.02,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA', 'sec':'dend', 'loc':0.5}
netParams.connParams['EML->EML'] = {
        'preConds': {'pop': 'EML'},
        'postConds': {'pop': 'EML'},
        'probability': 0.02,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA', 'sec':'dend', 'loc':0.5}
netParams.connParams['EMR->EMR'] = {
        'preConds': {'pop': 'EMR'},
        'postConds': {'pop': 'EMR'},
        'probability': 0.02,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA', 'sec':'dend', 'loc':0.5}
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
netParams.connParams['EV4->IV4'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'IV4'},
        'connList': blistEV4toIV4,
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
netParams.connParams['EIT->IIT'] = {
        'preConds': {'pop': 'EIT'},
        'postConds': {'pop': 'IIT'},
        'connList': blistEITtoIIT,
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
netParams.connParams['EML->IM'] = {
        'preConds': {'pop': 'EML'},
        'postConds': {'pop': 'IM'},
        #'probability': 0.25,
        'convergence': 25,
        'weight': 0.02 * cfg.EIGain,
        'delay': 2,
        'synMech': 'AMPA', 'sec':'soma', 'loc':0.5}
netParams.connParams['EMR->IM'] = {
        'preConds': {'pop': 'EMR'},
        'postConds': {'pop': 'IM'},
        #'probability': 0.25,
        'convergence': 25,
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
netParams.connParams['IV4->EV4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'EV4'},
        'connList': blistIV4toEV4,
        'weight': 0.02 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}
netParams.connParams['IIT->EIT'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'EIT'},
        'connList': blistIITtoEIT,
        'weight': 0.02 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}
netParams.connParams['IM->EML'] = {
        'preConds': {'pop': 'IM'},
        'postConds': {'pop': 'EML'},
        #'probability': 0.25,
        #'divergence': 9,
        'convergence': 13,
        'weight': 0.02 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}
netParams.connParams['IM->EMR'] = {
        'preConds': {'pop': 'IM'},
        'postConds': {'pop': 'EMR'},
        #'probability': 0.25,
        #'divergence': 9,
        'convergence': 13,
        'weight': 0.02 * cfg.IEGain,
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}


#I to I
netParams.connParams['IV1->IV1'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV1'},
        'probability': 0.25,
        'weight': 0.005 * cfg.IIGain, # 0.000
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}

netParams.connParams['IV4->IV4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IV4'},
        'probability': 0.25,
        'weight': 0.005 * cfg.IIGain, #0.000
        'delay': 2,
        'synMech': 'GABA', 'sec':'soma', 'loc':0.5}
netParams.connParams['IIT->IIT'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'IIT'},
        'probability': 0.25,
        'weight': 0.005 * cfg.IIGain, #0.000
        'delay': 2,
        'synMech': 'GABA' ,'sec':'soma', 'loc':0.5}
netParams.connParams['IM->IM'] = {
        'preConds': {'pop': 'IM'},
        'postConds': {'pop': 'IM'},
        'probability': 0.25,
        'weight': 0.005 * cfg.IIGain, #0.000
        'delay': 2,
        'synMech': 'GABA' ,'sec':'soma', 'loc':0.5}


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
netParams.connParams['EV4->EIT'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EIT'},
        'connList': blistEV4toEIT,
        #'convergence': 10,
        'weight': 0.01 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}
netParams.connParams['EIT->EML'] = {
        'preConds': {'pop': 'EIT'},
        'postConds': {'pop': 'EML'},
        #'connList': blistITtoMI,
        'convergence': 16,
        'weight': 0.0025 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL1},'sec':'dend', 'loc':0.5}
netParams.connParams['EIT->EMR'] = {
        'preConds': {'pop': 'EIT'},
        'postConds': {'pop': 'EMR'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0025 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL1},'sec':'dend', 'loc':0.5}

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
netParams.connParams['EV4->IIT'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'IIT'},
        'connList': blistEV4toIIT,
        #'convergence': 10,
        'weight': 0.00 * cfg.EIGain, #0.002
        'delay': 2,
        'synMech': 'AMPA','sec':'soma', 'loc':0.5}

#E to E feedbackward connections
netParams.connParams['EV1->ER'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'ER'},
        'connList': blistEV1toER,
        #'convergence': 10,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}
"""
netParams.connParams['EV4->EV1'] = {    # <<-- that's E -> I ?? or E -> E ?? weight is 0 but something wrong here
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EV1'},
        'connList': blistInV4toV1,
        #'convergence': 10,
        'weight': 0.000 * cfg.EEGain, #0.0001
        'delay': 2,
        'synMech': 'AMPA','sec':'dend', 'loc':0.5}
"""
netParams.connParams['EIT->EV4'] = {
        'preConds': {'pop': 'EIT'},
        'postConds': {'pop': 'EV4'},
        'connList': blistEITtoEV4,
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
netParams.connParams['IIT->EV4'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'EV4'},
        'connList': blistIITtoEV4,
        #'convergence': 10,
        'weight': 0.00 * cfg.IEGain, #0.002
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}

#I to I
netParams.connParams['IV1->IV4'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistIV1toIV4,
        #'convergence': 10,
        'weight': 0.00075 * cfg.IIGain, #0.00
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}
netParams.connParams['IV4->IIT'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IIT'},
        'connList': blistIV4toIIT,
        #'convergence': 10,
        'weight': 0.00075 * cfg.IIGain, #0.00
        'delay': 2,
        'synMech': 'GABA','sec':'soma', 'loc':0.5}

#Add direct connections from lower and higher visual areas to motor cortex
#Still no idea, how these connections should look like...just trying some numbers: 400 to 25 means convergence factor of 16
#AMPA
netParams.connParams['EV1->EMR'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'EMR'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0025 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL1},'sec':'dend', 'loc':0.5}
netParams.connParams['EV1->EML'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'EML'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0025 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL1},'sec':'dend', 'loc':0.5}
netParams.connParams['EV4->EMR'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EMR'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0025 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL1},'sec':'dend', 'loc':0.5}
netParams.connParams['EV4->EML'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EML'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0025 * cfg.EEGain,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL1},'sec':'dend', 'loc':0.5}

#NMDA
netParams.connParams['EV1N->EMRN'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'EMR'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.002 * cfg.EEGain,
        'delay': 2,
        'synMech': 'NMDA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL2},'sec':'dend', 'loc':0.5}
netParams.connParams['EV1N->EMLN'] = {
        'preConds': {'pop': 'EV1'},
        'postConds': {'pop': 'EML'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.002 * cfg.EEGain,
        'delay': 2,
        'synMech': 'NMDA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL2},'sec':'dend', 'loc':0.5}
netParams.connParams['EV4N->EMRN'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EMR'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.002 * cfg.EEGain,
        'delay': 2,
        'synMech': 'NMDA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL2},'sec':'dend', 'loc':0.5}
netParams.connParams['EV4N->EMLN'] = {
        'preConds': {'pop': 'EV4'},
        'postConds': {'pop': 'EML'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.002 * cfg.EEGain,
        'delay': 2,
        'synMech': 'NMDA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL2},'sec':'dend', 'loc':0.5}

###################################################################################################################################

sim.AIGame = None # placeholder

def recordAdjustableWeightsPop (sim, t, popname):
    if 'synweights' not in sim.simData: sim.simData['synweights'] = {sim.rank:[]}
    # record the plastic weights for specified popname
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids] # this is the set of MR cells
    for cell in lcell:
        for conn in cell.conns:
            if 'hSTDP' in conn:
                #print(conn.preGid, cell.gid, conn.synMech) #testing weight saving
                sim.simData['synweights'][sim.rank].append([t,conn.plast.params.RLon,conn.preGid,cell.gid,float(conn['hObj'].weight[0]),conn.synMech])
    return len(lcell)
                    
def recordAdjustableWeights (sim, t, lpop = ['EMR', 'EML']):
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

def saveGameBehavior(sim):
    with open(sim.ActionsRewardsfilename,'w') as fid3:
        for i in range(len(sim.allActions)):
            fid3.write('%0.1f' % sim.allTimes[i])
            fid3.write('\t%0.1f' % sim.allActions[i])
            fid3.write('\t%0.5f' % sim.allRewards[i])
            fid3.write('\t%0.1f' % sim.allProposedActions[i]) #the number of proposed action should be equal to the number of actions
            fid3.write('\n')

######################################################################################

def getFiringRatesWithInterval(trange = None, neuronal_pop = None):
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

InputImages = []
NBsteps = 0 # this is a counter for recording the plastic weights
epCount = []
last_obs = [] #make sure this does not introduce a bug
proposed_actions = [] 
last_ball_dir = 0 
total_hits = [] #number of times a ball is hit by racket as the ball changes its direction and player doesn't lose a score.
lSTDPmech = [] # global list of STDP mechanisms; so do not have to lookup at each interval function call 
    
def trainAgentFake(t):
    """ training interface between simulation and game environment
    """
    global NBsteps, last_obs, epCount, InputImages
    if t<21.0: # for the first time interval use first action randomly and other four actions based on relative position of ball and agent.
        last_obs = []
        if sim.rank == 0:
            rewards, actions, last_obs, epCount, InputImages = sim.AIGame.playGameFake(last_obs, epCount, InputImages)
    else: #the actions are generated based on relative positions of ball and Agent.
        if sim.rank == 0:
            rewards, actions, last_obs, epCount, InputImages = sim.AIGame.playGameFake(last_obs, epCount, InputImages)
    #print('actions generated by model are: ', actions)
    F_R1 = getFiringRatesWithInterval([t-100,t-80], sim.net.pops['EMR'].cellGids)
    F_R2 = getFiringRatesWithInterval([t-80,t-60], sim.net.pops['EMR'].cellGids)
    F_R3 = getFiringRatesWithInterval([t-60,t-40], sim.net.pops['EMR'].cellGids)
    F_R4 = getFiringRatesWithInterval([t-40,t-20], sim.net.pops['EMR'].cellGids)
    F_R5 = getFiringRatesWithInterval([t-20,t], sim.net.pops['EMR'].cellGids)
    F_L1 = getFiringRatesWithInterval([t-100,t-80], sim.net.pops['EML'].cellGids)
    F_L2 = getFiringRatesWithInterval([t-80,t-60], sim.net.pops['EML'].cellGids)
    F_L3 = getFiringRatesWithInterval([t-60,t-40], sim.net.pops['EML'].cellGids)
    F_L4 = getFiringRatesWithInterval([t-40,t-20], sim.net.pops['EML'].cellGids)
    F_L5 = getFiringRatesWithInterval([t-20,t], sim.net.pops['EML'].cellGids)
    fid4.write('%0.1f' % t)
    fid4.write('\t%0.1f' % F_R1)
    fid4.write('\t%0.1f' % F_R2)
    fid4.write('\t%0.1f' % F_R3)
    fid4.write('\t%0.1f' % F_R4)
    fid4.write('\t%0.1f' % F_R5)
    fid4.write('\t%0.1f' % F_L1)
    fid4.write('\t%0.1f' % F_L2)
    fid4.write('\t%0.1f' % F_L3)
    fid4.write('\t%0.1f' % F_L4)
    fid4.write('\t%0.1f' % F_L5)
    fid4.write('\n')
    critic = sum(rewards) # get critic signal (-1, 0 or 1)
    if critic>0:
        critic = 1
    elif critic<0:
        critic = -1
    else:
        critic = 0
    if critic != 0: # if critic signal indicates punishment (-1) or reward (+1)
        if sim.rank==0: print('t=',t,'- adjusting weights based on RL critic value:', critic)
        for STDPmech in lSTDPmech: STDPmech.reward_punish(float(critic))
    print('rewards are : ', rewards)
    for action in actions:
        sim.allActions.append(action)
    for reward in rewards:
        sim.allRewards.append(reward)
    ltpnt = t-20
    for _ in range(5):
        ltpnt = ltpnt+4
        sim.allTimes.append(ltpnt)
    if sim.rank == 0:
        sim.AIGame.run(t,sim)
    print('trainAgent time is : ', t)
    NBsteps = NBsteps+1
    if NBsteps % recordWeightStepSize == 0:
        #if t%recordWeightDT==0:
        print('Weights Recording Time:', t) 
        recordWeights(sim, t)

def updateInputRates ():
    # update the source firing rates for the R neuron population, based on image contents
    #also update the firing rates for the direction sensitive neurons based on image contents
    if sim.rank == 0:
        if dconf['verbose'] > 1:
          print(sim.rank,'broadcasting firing rates:',np.where(sim.AIGame.firing_rates==np.amax(sim.AIGame.firing_rates)),np.amax(sim.AIGame.firing_rates))        
        sim.pc.broadcast(sim.AIGame.fvec.from_python(sim.AIGame.firing_rates),0)
        firing_rates = sim.AIGame.firing_rates
        sim.pc.broadcast(sim.AIGame.fvecR.from_python(sim.AIGame.directionsR),0)
        firing_rates_dirR = sim.AIGame.directionsR
        #print('Firing Rates of R-DIR:',firing_rates_dirR)
        sim.pc.broadcast(sim.AIGame.fvecL.from_python(sim.AIGame.directionsL),0)
        firing_rates_dirL = sim.AIGame.directionsL
        sim.pc.broadcast(sim.AIGame.fvecU.from_python(sim.AIGame.directionsUp),0)
        firing_rates_dirUp = sim.AIGame.directionsUp
        sim.pc.broadcast(sim.AIGame.fvecD.from_python(sim.AIGame.directionsDown),0)
        firing_rates_dirDown = sim.AIGame.directionsDown
    else:
        fvec = h.Vector()
        sim.pc.broadcast(fvec,0)
        firing_rates = fvec.to_python()
        fvecR = h.Vector()
        sim.pc.broadcast(fvecR,0)
        firing_rates_dirR = fvecR.to_python()
        print(sim.rank,'recrived firing Rates of R-DIR:',firing_rates_dirR)
        fvecL = h.Vector()
        sim.pc.broadcast(fvecL,0)
        firing_rates_dirL = fvecL.to_python()
        fvecU = h.Vector()
        sim.pc.broadcast(fvecU,0)
        firing_rates_dirUp = fvecU.to_python()
        fvecD = h.Vector()
        sim.pc.broadcast(fvecD,0)
        firing_rates_dirDown = fvecD.to_python()
        if dconf['verbose'] > 1:
          print(sim.rank,'received firing rates:',np.where(firing_rates==np.amax(firing_rates)),np.amax(firing_rates))
          print(sim.rank,'received R-firing rates:',np.where(firing_rates_dirR==np.amax(firing_rates_dirR)),np.amax(firing_rates_dirR))
          print(sim.rank,'received L-firing rates:',np.where(firing_rates_dirL==np.amax(firing_rates_dirL)),np.amax(firing_rates_dirL))
          print(sim.rank,'received U-firing rates:',np.where(firing_rates_dirUp==np.amax(firing_rates_dirUp)),np.amax(firing_rates_dirUp))
          print(sim.rank,'received D-firing rates:',np.where(firing_rates_dirDown==np.amax(firing_rates_dirDown)),np.amax(firing_rates_dirDown))

    # update input firing rates for stimuli to R cells
    lRcell = [c for c in sim.net.cells if c.gid in sim.net.pops['ER'].cellGids] # this is the set of R cells
    R_offset = 0 #np.amin(sim.net.pops['ER'].cellGids)
    print('R offset:', R_offset)
    if dconf['verbose'] > 1: print(sim.rank,'updating len(lRcell)=',len(lRcell),'source firing rates. len(firing_rates)=',len(firing_rates))
    for cell in lRcell:  
        for stim in cell.stims:
            if stim['source'] == 'stimMod':
                stim['hObj'].interval = 10 #1000.0/firing_rates[int(cell.gid)]
                #print('cell GID: ', int(cell.gid), 'vs cell ID with offset: ', int(cell.gid-R_offset)) # interval in ms as a function of rate; is cell.gid correct index???
    # update input firing rates for stimuli to R-direction cells
    lRDircell = [c for c in sim.net.cells if c.gid in sim.net.pops['EV1D0'].cellGids] # this is the set of 0-degree direction selective cells
    RDir_offset = 900 #np.amin(sim.net.pops['EV1D0'].cellGids) #hard coded number--change later
    if dconf['verbose'] > 1: print(sim.rank,'updating len(lRDircell)=',len(lRDircell),'source firing rates. len(firing_rates_dirR)=',len(firing_rates_dirR))
    for cell in lRDircell:  
        for stim in cell.stims:
            if stim['source'] == 'stimMod':
                stim['hObj'].interval = 10
                print('R-Dir Neurons:',sim.net.pops['EV1D0'].cellGids, 'cells:', lRDircell,'Neuron: ', cell, 'cell gid: ', cell.gid)
                #stim['hObj'].interval = 1000.0/firing_rates_dirR[int(cell.gid-RDir_offset)] # interval in ms as a function of rate; is cell.gid correct index???
                #print('Neuron ', cell, 'on', sim.rank,' was assigned ISI of ', stim['hObj'].interval, ' ms')
    # update input firing rates for stimuli to L-direction cells
    lLDircell = [c for c in sim.net.cells if c.gid in sim.net.pops['EV1D180'].cellGids] # this is the set of 180-degree direction selective cells
    LDir_offset = np.amin(sim.net.pops['EV1D180'].cellGids)
    if dconf['verbose'] > 1: print(sim.rank,'updating len(lLDircell)=',len(lLDircell),'source firing rates. len(firing_rates_dirL)=',len(firing_rates_dirL))
    for cell in lLDircell:  
        for stim in cell.stims:
            if stim['source'] == 'stimMod':
                stim['hObj'].interval = 1000.0/firing_rates_dirL[int(cell.gid-LDir_offset)] # interval in ms as a function of rate; is cell.gid correct index???
    # update input firing rates for stimuli to Up-direction cells
    lUDircell = [c for c in sim.net.cells if c.gid in sim.net.pops['EV1D90'].cellGids] # this is the set of 90-degree direction selective cells
    UDir_offset = np.amin(sim.net.pops['EV1D90'].cellGids)
    if dconf['verbose'] > 1: print(sim.rank,'updating len(lUDircell)=',len(lUDircell),'source firing rates. len(firing_rates_dirUp)=',len(firing_rates_dirUp))
    for cell in lUDircell:  
        for stim in cell.stims:
            if stim['source'] == 'stimMod':
                stim['hObj'].interval = 1000.0/firing_rates_dirUp[int(cell.gid-UDir_offset)] # interval in ms as a function of rate; is cell.gid correct index???
    # update input firing rates for stimuli to Down-direction cells
    lDDircell = [c for c in sim.net.cells if c.gid in sim.net.pops['EV1D270'].cellGids] # this is the set of 270-degree direction selective cells
    DDir_offset = np.amin(sim.net.pops['EV1D270'].cellGids)
    if dconf['verbose'] > 1: print(sim.rank,'updating len(lDDircell)=',len(lDDircell),'source firing rates. len(firing_rates_dirDown)=',len(firing_rates_dirDown))
    for cell in lDDircell:  
        for stim in cell.stims:
            if stim['source'] == 'stimMod':
                stim['hObj'].interval = 1000.0/firing_rates_dirDown[int(cell.gid-DDir_offset)] # interval in ms as a function of rate; is cell.gid correct index???

def trainAgent (t):
    """ training interface between simulation and game environment
    """
    global NBsteps, epCount, InputImages, last_obs, proposed_actions, last_ball_dir, total_hits
    vec = h.Vector()
    if t<100.0: # for the first time interval use randomly selected actions
        actions =[]
        for _ in range(5):
            action = dconf['movecodes'][random.randint(0,len(dconf['movecodes'])-1)]
            actions.append(action)
    else: #the actions should be based on the activity of motor cortex (MO) 1085-1093
        F_R1 = getFiringRatesWithInterval([t-100,t-80], sim.net.pops['EMR'].cellGids) 
        sim.pc.allreduce(vec.from_python([F_R1]), 1) # sum
        F_R1 = vec.to_python()[0] 
        F_R2 = getFiringRatesWithInterval([t-80,t-60], sim.net.pops['EMR'].cellGids)
        sim.pc.allreduce(vec.from_python([F_R2]), 1) # sum
        F_R2 = vec.to_python()[0] 
        F_R3 = getFiringRatesWithInterval([t-60,t-40], sim.net.pops['EMR'].cellGids)
        sim.pc.allreduce(vec.from_python([F_R3]), 1) # sum
        F_R3 = vec.to_python()[0] 
        F_R4 = getFiringRatesWithInterval([t-40,t-20], sim.net.pops['EMR'].cellGids)
        sim.pc.allreduce(vec.from_python([F_R4]), 1) # sum
        F_R4 = vec.to_python()[0] 
        F_R5 = getFiringRatesWithInterval([t-20,t], sim.net.pops['EMR'].cellGids)
        sim.pc.allreduce(vec.from_python([F_R5]), 1) # sum
        F_R5 = vec.to_python()[0] 
        F_L1 = getFiringRatesWithInterval([t-100,t-80], sim.net.pops['EML'].cellGids) 
        sim.pc.allreduce(vec.from_python([F_L1]), 1) # sum
        F_L1 = vec.to_python()[0] 
        F_L2 = getFiringRatesWithInterval([t-80,t-60], sim.net.pops['EML'].cellGids)
        sim.pc.allreduce(vec.from_python([F_L2]), 1) # sum
        F_L2 = vec.to_python()[0] 
        F_L3 = getFiringRatesWithInterval([t-60,t-40], sim.net.pops['EML'].cellGids)
        sim.pc.allreduce(vec.from_python([F_L3]), 1) # sum
        F_L3 = vec.to_python()[0] 
        F_L4 = getFiringRatesWithInterval([t-40,t-20], sim.net.pops['EML'].cellGids)
        sim.pc.allreduce(vec.from_python([F_L4]), 1) # sum
        F_L4 = vec.to_python()[0] 
        F_L5 = getFiringRatesWithInterval([t-20,t], sim.net.pops['EML'].cellGids)
        sim.pc.allreduce(vec.from_python([F_L5]), 1) # sum
        F_L5 = vec.to_python()[0] 
        if sim.rank==0:
            print('Firing rates: ', F_R1, F_R2, F_R3, F_R4, F_R5, F_L1, F_L2, F_L3, F_L4, F_L5)
            fid4.write('%0.1f' % t)
            fid4.write('\t%0.1f' % F_R1)
            fid4.write('\t%0.1f' % F_R2)
            fid4.write('\t%0.1f' % F_R3)
            fid4.write('\t%0.1f' % F_R4)
            fid4.write('\t%0.1f' % F_R5)
            fid4.write('\t%0.1f' % F_L1)
            fid4.write('\t%0.1f' % F_L2)
            fid4.write('\t%0.1f' % F_L3)
            fid4.write('\t%0.1f' % F_L4)
            fid4.write('\t%0.1f' % F_L5)
            fid4.write('\n')
            actions = []
            if F_R1>F_L1:
                actions.append(dconf['moves']['UP']) #UP
            elif F_R1<F_L1:
                actions.append(dconf['moves']['DOWN']) # Down
            else:
                actions.append(dconf['moves']['NOMOVE']) # No move 
            if F_R2>F_L2:
                actions.append(dconf['moves']['UP']) #UP
            elif F_R2<F_L2:
                actions.append(dconf['moves']['DOWN']) #Down
            else:
                actions.append(dconf['moves']['NOMOVE']) #No move
            if F_R3>F_L3:
                actions.append(dconf['moves']['UP']) #UP
            elif F_R3<F_L3:
                actions.append(dconf['moves']['DOWN']) #Down
            else:
                actions.append(dconf['moves']['NOMOVE']) #No move
            if F_R4>F_L4:
                actions.append(dconf['moves']['UP']) #UP
            elif F_R4<F_L4:
                actions.append(dconf['moves']['DOWN']) #Down
            else:
                actions.append(dconf['moves']['NOMOVE']) #No move
            if F_R5>F_L5:
                actions.append(dconf['moves']['UP']) #UP
            elif F_R5<F_L5:
                actions.append(dconf['moves']['DOWN']) #Down
            else:
                actions.append(dconf['moves']['NOMOVE']) #No move
            #print('actions generated by model are: ', actions)
            # actions = [dconf['moves']['UP'] for i in range(5)] # force move UP for testing            
    if sim.rank == 0:
        print('Model actions:', actions)
        rewards, epCount, InputImages, last_obs, proposed_actions, last_ball_dir, total_hits = sim.AIGame.playGame(actions, epCount, InputImages, last_obs, last_ball_dir)
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
    else: # other workers
        sim.pc.broadcast(vec, 0) # receive critic value from master node
        critic = vec.to_python()[0] # critic is first element of the array
    if critic != 0: # if critic signal indicates punishment (-1) or reward (+1)
        if sim.rank==0: print('t=',t,'- adjusting weights based on RL critic value:', critic)
        for STDPmech in lSTDPmech: STDPmech.reward_punish(float(critic))        
    if sim.rank==0:
        print('Game rewards:', rewards) # only rank 0 has access to rewards      
        for action in actions:
            sim.allActions.append(action)
        for pactions in proposed_actions: #also record proposed actions
            sim.allProposedActions.append(pactions)
        for reward in rewards: # this generates an error - since rewards only declared for sim.rank==0; bug?
            sim.allRewards.append(reward)
        for ltpnt in [t-80, t-60, t-40, t-20, t-0]: sim.allTimes.append(ltpnt)
    updateInputRates() # update firing rate of inputs to R population (based on image content)                
    NBsteps = NBsteps+1
    if NBsteps % recordWeightStepSize == 0:
        #if t%recordWeightDT==0:
        if dconf['verbose'] > 0 and sim.rank==0:
            print('Weights Recording Time:', t, 'NBsteps:',NBsteps,'recordWeightStepSize:',recordWeightStepSize)
        recordAdjustableWeights(sim, t) 
        #recordWeights(sim, t)

def getAllSTDPObjects (sim):
  # get all the STDP objects from the simulation's cells
  lSTDPmech = []
  for cell in sim.net.cells:
      for conn in cell.conns:
          STDPmech = conn.get('hSTDP')  # check if has STDP mechanism
          if STDPmech:   # make sure it is not None
            lSTDPmech.append(STDPmech)
  return lSTDPmech
        
#Alterate to create network and run simulation
# create network object and set cfg and net params; pass simulation config and network params as arguments
sim.initialize(simConfig = simConfig, netParams = netParams)
sim.net.createPops()                      # instantiate network populations
sim.net.createCells()                     # instantiate network cells based on defined populations
sim.net.connectCells()                    # create connections between cells based on params
sim.net.addStims()                      #instantiate netStim
sim.setupRecording()                  # setup variables to record for each cell (spikes, V traces, etc)

lSTDPmech = getAllSTDPObjects(sim) # get all the STDP objects up-front

def updateSTDPWeights (sim, W):
    #this function assign weights stored in 'ResumeSimFromFile' to all connections by matching pre and post neuron ids  
    # get all the simulation's cells (on a given node)
    for cell in sim.net.cells:
        cpostID = cell.gid#find postID
        for conn in cell.conns:
            cpreID = conn.preGid  #find preID
            cConnW = W[(W.postid==cpostID) & (W.preid==cpreID)] #find the record for a connection with pre and post neuron ID
            #find weight for the STDP connection between preID and postID
            for idx in cConnW.index: 
                cW = cConnW.at[idx,'weight']
                cstdp = cConnW.at[idx,'stdptype'] 
                #STDPmech = conn.get('hSTDP')  # check if has STDP mechanism
                if dconf['verbose'] > 1:
                    print('weight updated:', cW, cstdp)
                if cstdp:   # make sure it is not None
                    conn['hObj'].weight[0] = cW

#if specified 'ResumeSim' = 1, load the connection data from 'ResumeSimFromFile' and assign weights to STDP synapses  
if dconf['simtype']['ResumeSim']:
    try:
        from simdat import readinweights
        A = readinweights(pickle.load(open(dconf['simtype']['ResumeSimFromFile'],'rb')))
        updateSTDPWeights(sim, A[A.time == max(A.time)]) # take the latest weights saved
        if sim.rank==0: print('Updated STDP weights')
    except:
        print('Could not restore STDP weights from file.')

if sim.rank == 0: 
    from aigame import AIGame
    sim.AIGame = AIGame() # only create AIGame on node 0
    # node 0 saves the json config file
    # this is just a precaution since simConfig pkl file has MOST of the info; ideally should adjust simConfig to contain ALL of the required info
    from utils import backupcfg
    backupcfg(dconf['sim']['name']) 
    
sim.runSimWithIntervalFunc(100.0,trainAgent) # has periodic callback to adjust STDP weights based on RL signal
sim.gatherData() # gather data from different nodes
sim.saveData() # save data to disk
if sim.rank==0:
    print('SAVING RASTER DATA')
    sim.analysis.plotRaster(include = ['allCells'],saveData = dconf['sim']['name']+'raster.pkl',showFig=True)
    sim.analysis.plotData()
#if sim.rank == 0: # only rank 0 should save. otherwise all the other nodes could over-write the output or quit first; rank 0 plots
#    if dconf['sim']['doplot']:
#        print('plot raster:')
#        sim.analysis.plotRaster()
#        sim.analysis.plotData()    
#        if sim.plotWeights: plotWeights() 
#    if sim.saveWeights:
#        #saveWeights(sim, recordWeightDCells)
#        saveGameBehavior(sim)
#        fid5 = open('data/'+dconf['sim']['name']+'ActionsPerEpisode.txt','w')
#        for i in range(len(epCount)):
#            fid5.write('\t%0.1f' % epCount[i])
#            fid5.write('\n')
#    if sim.saveInputImages:
#        InputImages = np.array(InputImages)
#        print(InputImages.shape)  
#        with open('data/'+dconf['sim']['name']+'InputImages.txt', 'w') as outfile:
#            outfile.write('# Array shape: {0}\n'.format(InputImages.shape))
#            for Input_Image in InputImages:
#                np.savetxt(outfile, Input_Image, fmt='%-7.2f')
#                outfile.write('# New slice\n')
#    if dconf['sim']['doquit']: quit()
