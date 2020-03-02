from netpyne import specs, sim
from neuron import h
import numpy as np
import random
from conf import dconf # configuration dictionary
import pandas as pd

random.seed(1234) # this will not work properly across runs with different number of nodes

sim.allTimes = []
sim.allRewards = [] # list to store all rewards
sim.allActions = [] # list to store all actions
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

ETypes = ['E','EV1','EV4','EIT', 'EML', 'EMR']
ITypes = ['InR','InV1','InV4','InIT']

NB_Rneurons = dconf['net']['E'] * scale
NB_V1neurons = dconf['net']['EV1'] * scale
NB_IRneurons = dconf['net']['InR'] * scale
NB_IV1neurons = dconf['net']['InV1'] * scale
NB_V4neurons = dconf['net']['EV4'] * scale
NB_ITneurons = dconf['net']['EIT'] * scale
NB_IV4neurons = dconf['net']['InV4'] * scale
NB_IITneurons = dconf['net']['InIT'] * scale
NB_MLneurons = dconf['net']['ML'] * scale
NB_MRneurons = dconf['net']['MR'] * scale

# Network parameters
netParams = specs.NetParams() #object of class NetParams to store the network parameters

#Population parameters
netParams.popParams['R'] = {'cellType': 'E', 'numCells': NB_Rneurons, 'cellModel': 'HH'}  #6400 neurons to represent 6400 pixels, now we have 400 pixels
netParams.popParams['IR'] = {'cellType': 'InR', 'numCells': NB_IRneurons, 'cellModel': 'HH'}
netParams.popParams['V1'] = {'cellType': 'EV1', 'numCells': NB_V1neurons, 'cellModel': 'HH'} #6400 neurons
netParams.popParams['IV1'] = {'cellType': 'InV1', 'numCells': NB_IV1neurons, 'cellModel': 'HH'} #1600
netParams.popParams['V4'] = {'cellType': 'EV4', 'numCells': NB_V4neurons, 'cellModel': 'HH'} #1600 neurons
netParams.popParams['IV4'] = {'cellType': 'InV4', 'numCells': NB_IV4neurons, 'cellModel': 'HH'} #400
netParams.popParams['IT'] = {'cellType': 'EIT', 'numCells': NB_ITneurons, 'cellModel': 'HH'} #400 neurons
netParams.popParams['IIT'] = {'cellType': 'InIT', 'numCells': NB_IITneurons, 'cellModel': 'HH'} #100
netParams.popParams['ML'] = {'cellType': 'EML', 'numCells': NB_MLneurons, 'cellModel': 'HH'} #400
netParams.popParams['MR'] = {'cellType': 'EMR', 'numCells': NB_MRneurons, 'cellModel': 'HH'} #100

netParams.cellParams['ERule'] = {               # cell rule label
        'conds': {'cellType': ETypes},              #properties will be applied to cells that match these conditions
        'secs': {'soma':                        #sections
                {'geom': {'diam':10, 'L':10, 'Ra':120},         #geometry
                'mechs': {'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}}}    #mechanism

netParams.cellParams['IRule'] = {               # cell rule label
        'conds': {'cellType': ITypes},              #properties will be applied to cells that match these conditions
        'secs': {'soma':                        #sections
                {'geom': {'diam':10, 'L':10, 'Ra':120},         #geometry
                'mechs': {'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}}}    #mechanism

## Synaptic mechanism parameters
netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}  # excitatory synaptic mechanism
netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80}  # inhibitory synaptic mechanism

#wmin should be set to the initial/baseline weight of the connection.
STDPparams = {'hebbwt': 0.0001, 'antiwt':-0.00001, 'wbase': 0.0012, 'wmax': 50, 'RLon': 0 , 'RLhebbwt': 0.001, 'RLantiwt': -0.000,
        'tauhebb': 10, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 0, 'verbose':0}

STDPparamsRL = {'hebbwt': 0.0000, 'antiwt':-0.0000, 'wbase': 0.0005, 'wmax': 0.1, 'RLon': 1 , 'RLhebbwt': 0.0001, 'RLantiwt': -0.000,
                'tauhebb': 10, 'RLlenhebb': 800 ,'RLlenanti': 100, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 0, 'verbose':0}

# these are the image-based inputs provided to the R (retinal) cells
netParams.stimSourceParams['stimMod'] = {'type': 'NetStim', 'rate': 'variable', 'noise': 0}
netParams.stimTargetParams['stimMod->all'] = {'source': 'stimMod',
        'conds': {'pop': 'R'},
        'convergence': 1,
        'weight': 0.01,
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
netParams.stimTargetParams['bkg->all'] = {'source': 'bkg', 'conds': {'cellType': ['InR','InV1','InV4','InIT']}, 'weight': 0.0, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'}

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
blistEtoV1 = connectLayerswithOverlap(NBpreN = NB_Rneurons, NBpostN = NB_V1neurons, overlap_xdir = 3)
blistV1toV4 = connectLayerswithOverlap(NBpreN = NB_V1neurons, NBpostN = NB_V4neurons, overlap_xdir = 3)
blistV4toIT = connectLayerswithOverlap(NBpreN = NB_V4neurons, NBpostN = NB_ITneurons, overlap_xdir = 3) #was 15
#blistITtoMI = connectLayerswithOverlap(NBpreN = NB_ITneurons, NBpostN = NB_MIneurons, overlap_xdir = 3) #Not sure if this is a good strategy instead of all to all
#blistMItoMO = connectLayerswithOverlap(NBpreN = NB_MIneurons, NBpostN = NB_MOneurons, overlap_xdir = 3) #was 19
#blistMItoMO: Feedforward for MI to MO is all to all and can be specified in the connection statement iteself

#E to I - Feedforward connections
blistEtoInV1 = connectLayerswithOverlap(NBpreN = NB_Rneurons, NBpostN = NB_IV1neurons, overlap_xdir = 3)
blistV1toInV4 = connectLayerswithOverlap(NBpreN = NB_V1neurons, NBpostN = NB_IV4neurons, overlap_xdir = 3) #was 15
blistV4toInIT = connectLayerswithOverlap(NBpreN = NB_V4neurons, NBpostN = NB_IITneurons, overlap_xdir = 3) #was 15

#E to I - WithinLayer connections
blistRtoInR = connectLayerswithOverlap(NBpreN = NB_Rneurons, NBpostN = NB_IRneurons, overlap_xdir = 3)
blistV1toInV1 = connectLayerswithOverlap(NBpreN = NB_V1neurons, NBpostN = NB_IV1neurons, overlap_xdir = 3)
blistV4toInV4 = connectLayerswithOverlap(NBpreN = NB_V4neurons, NBpostN = NB_IV4neurons, overlap_xdir = 3)
blistITtoInIT = connectLayerswithOverlap(NBpreN = NB_ITneurons, NBpostN = NB_IITneurons, overlap_xdir = 3)

#I to E - WithinLayer Inhibition
blistInRtoR = connectLayerswithOverlapDiv(NBpreN = NB_IRneurons, NBpostN = NB_Rneurons, overlap_xdir = 5)
blistInV1toV1 = connectLayerswithOverlapDiv(NBpreN = NB_IV1neurons, NBpostN = NB_V1neurons, overlap_xdir = 5)
blistInV4toV4 = connectLayerswithOverlapDiv(NBpreN = NB_IV4neurons, NBpostN = NB_V4neurons, overlap_xdir = 5)
blistInITtoIT = connectLayerswithOverlapDiv(NBpreN = NB_IITneurons, NBpostN = NB_ITneurons, overlap_xdir = 5)

#Feedbackward excitation
#E to E  
blistV1toE = connectLayerswithOverlapDiv(NBpreN = NB_V1neurons, NBpostN = NB_Rneurons, overlap_xdir = 3)
blistV4toV1 = connectLayerswithOverlapDiv(NBpreN = NB_V4neurons, NBpostN = NB_V1neurons, overlap_xdir = 3)
blistITtoV4 = connectLayerswithOverlapDiv(NBpreN = NB_ITneurons, NBpostN = NB_V4neurons, overlap_xdir = 3)

#Feedforward inhibition
#I to I
blistInV1toInV4 = connectLayerswithOverlap(NBpreN = NB_IV1neurons, NBpostN = NB_IV4neurons, overlap_xdir = 5)
blistInV4toInIT = connectLayerswithOverlap(NBpreN = NB_IV4neurons, NBpostN = NB_IITneurons, overlap_xdir = 5)

#Feedbackward inhibition
#I to E 
blistInV1toE = connectLayerswithOverlapDiv(NBpreN = NB_IV1neurons, NBpostN = NB_Rneurons, overlap_xdir = 5)
blistInV4toV1 = connectLayerswithOverlapDiv(NBpreN = NB_IV4neurons, NBpostN = NB_V1neurons, overlap_xdir = 5)
blistInITtoV4 = connectLayerswithOverlapDiv(NBpreN = NB_IITneurons, NBpostN = NB_V4neurons, overlap_xdir = 5)

#Local excitation
#E to E
netParams.connParams['R->R'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'R'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['V1->V1'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'V1'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['V4->V4'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'V4'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['IT->IT'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'IT'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['ML->ML'] = {
        'preConds': {'pop': 'ML'},
        'postConds': {'pop': 'ML'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['MR->MR'] = {
        'preConds': {'pop': 'MR'},
        'postConds': {'pop': 'MR'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}        
#E to I
netParams.connParams['R->IR'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'IR'},
        'connList': blistRtoInR,
        #'probability': 0.23,
        #'convergence': 9,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['V1->IV1'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'IV1'},
        'connList': blistV1toInV1,
        #'probability': 0.23,
        #'convergence': 9,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['V4->IV4'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'IV4'},
        'connList': blistV4toInV4,
        #'probability': 0.23,
        #'convergence': 9,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['IT->IIT'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'IIT'},
        'connList': blistITtoInIT,
        #'probability': 0.23,
        #'convergence': 9,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'AMPA'}

#Local inhibition
#I to E
netParams.connParams['IR->R'] = {
        'preConds': {'pop': 'IR'},
        'postConds': {'pop': 'R'},
        'connList': blistInRtoR,
        #'probability': 0.02,
        #'divergence': 9,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'GABA'}
netParams.connParams['IV1->V1'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'V1'},
        'connList': blistInV1toV1,
        #'probability': 0.02,
        #'divergence': 9,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'GABA'}
netParams.connParams['IV4->V4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'V4'},
        'connList': blistInV4toV4,
        #'probability': 0.02,
        #'divergence': 9,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'GABA'}
netParams.connParams['IIT->IT'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'IT'},
        'connList': blistInITtoIT,
        #'probability': 0.02,
        #'divergence': 9,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'GABA'}

#I to I
netParams.connParams['IV1->IV1'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV1'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'GABA'}

netParams.connParams['IV4->IV4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IV4'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'GABA'}
netParams.connParams['IIT->IIT'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'IIT'},
        'probability': 0.02,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'GABA'}

#E to E feedforward connections
netParams.connParams['R->V1'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'V1'},
        'connList': blistEtoV1,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 2,
        'synMech': 'AMPA'}

netParams.connParams['V1->V4'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'V4'},
        'connList': blistV1toV4,
        #'convergence': 10,
        'weight': 0.001,
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['V4->IT'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'IT'},
        'connList': blistV4toIT,
        #'convergence': 10,
        'weight': 0.001,
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['IT->ML'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'ML'},
        #'connList': blistITtoMI,
        'convergence': 16,
        #'weight': 0.0025,
        'weight': 0.0005,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}
netParams.connParams['IT->MR'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'MR'},
        #'connList': blistMItoMO,
        'convergence': 16,
        #'weight': 0.0025,
        'weight': 0.0005,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}

#E to I feedforward connections
netParams.connParams['R->IV1'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'IV1'},
        'connList': blistEtoInV1,
        #'convergence': 10,
        'weight': 0.00, #0.002
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['V1->IV4'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistV1toInV4,
        #'convergence': 10,
        'weight': 0.00, #0.002
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['V4->IIT'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'IIT'},
        'connList': blistV4toInIT,
        #'convergence': 10,
        'weight': 0.00, #0.002
        'delay': 2,
        'synMech': 'AMPA'}

#E to E feedbackward connections
netParams.connParams['V1->R'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'R'},
        'connList': blistV1toE,
        #'convergence': 10,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['V4->V1'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'V1'},
        'connList': blistInV4toV1,
        #'convergence': 10,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}
netParams.connParams['IT->V4'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'V4'},
        'connList': blistITtoV4,
        #'convergence': 10,
        'weight': 0.000, #0.0001
        'delay': 2,
        'synMech': 'AMPA'}

#I to E feedbackward connections
netParams.connParams['IV1->R'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'R'},
        'connList': blistInV1toE,
        #'convergence': 10,
        'weight': 0.00, #0.002
        'delay': 2,
        'synMech': 'GABA'}
netParams.connParams['IV4->V1'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'V1'},
        'connList': blistInV4toV1,
        #'convergence': 10,
        'weight': 0.00, #0.002
        'delay': 2,
        'synMech': 'GABA'}
netParams.connParams['IIT->V4'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'V4'},
        'connList': blistInITtoV4,
        #'convergence': 10,
        'weight': 0.00, #0.002
        'delay': 2,
        'synMech': 'GABA'}

#I to I
netParams.connParams['IV1->IV4'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistInV1toInV4,
        #'convergence': 10,
        'weight': 0.00, #0.002
        'delay': 2,
        'synMech': 'GABA'}
netParams.connParams['IV4->IIT'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IIT'},
        'connList': blistInV4toInIT,
        #'convergence': 10,
        'weight': 0.00, #0.002
        'delay': 2,
        'synMech': 'GABA'}

#Add direct connections from lower and higher visual areas to motor cortex
#Still no idea, how these connections should look like...just trying some numbers: 400 to 25 means convergence factor of 16
netParams.connParams['V1->MR'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'MR'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0005,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}
netParams.connParams['V1->ML'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'ML'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0005,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}
netParams.connParams['V4->MR'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'MR'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0005,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}
netParams.connParams['V4->ML'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'ML'},
        #'connList': blistMItoMO,
        'convergence': 16,
        'weight': 0.0005,
        'delay': 2,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}

#Simulation options
simConfig = specs.SimConfig()           # object of class SimConfig to store simulation configuration

simConfig.duration = dconf['sim']['duration'] # 100e3 # 0.1e5                      # Duration of the simulation, in ms
simConfig.dt = dconf['sim']['dt']                            # Internal integration timestep to use
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
simConfig.analysis['plotTraces'] = {'include': [1159, 1169, 1179, 1189, 1199]}
#simConfig.analysis['plotRaster'] = {'timeRange': [500,1000],'popRates':'overlay','saveData':'data/RasterData.pkl','showFig':True}
simConfig.analysis['plotRaster'] = {'popRates':'overlay','saveData':'data/'+dconf['sim']['name']+'RasterData.pkl','showFig':True}
#simConfig.analysis['plot2Dnet'] = True 
#simConfig.analysis['plotConn'] = True           # plot connectivity matrix
###################################################################################################################################

sim.AIGame = None # placeholder

def recordAdjustableWeightsPop (sim, t, popname):
    if 'synweights' not in sim.simData: sim.simData['synweights'] = {sim.rank:[]}
    # record the plastic weights for specified popname
    lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids] # this is the set of MR cells
    for cell in lcell:
        for conn in cell.conns:
            if 'hSTDP' in conn:
                sim.simData['synweights'][sim.rank].append([t,conn.plast.params.RLon,conn.preGid,cell.gid,float(conn['hObj'].weight[0])])
    return len(lcell)
                    
def recordAdjustableWeights (sim, t, lpop = ['MR', 'ML']):
    """ record the STDP weights during the simulation - called in trainAgent
    """
    for pop in lpop: recordAdjustableWeightsPop(sim, t, pop)

def recordWeights (sim, t):
    """ record the STDP weights during the simulation - called in trainAgent
    """
    #lRcell = [c for c in sim.net.cells if c.gid in sim.net.pops['R'].cellGids]
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
            fid3.write('\t%0.1f' % sim.allRewards[i])
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
    F_R1 = getFiringRatesWithInterval([t-100,t-80], sim.net.pops['MR'].cellGids)
    F_R2 = getFiringRatesWithInterval([t-80,t-60], sim.net.pops['MR'].cellGids)
    F_R3 = getFiringRatesWithInterval([t-60,t-40], sim.net.pops['MR'].cellGids)
    F_R4 = getFiringRatesWithInterval([t-40,t-20], sim.net.pops['MR'].cellGids)
    F_R5 = getFiringRatesWithInterval([t-20,t], sim.net.pops['MR'].cellGids)
    F_L1 = getFiringRatesWithInterval([t-100,t-80], sim.net.pops['ML'].cellGids)
    F_L2 = getFiringRatesWithInterval([t-80,t-60], sim.net.pops['ML'].cellGids)
    F_L3 = getFiringRatesWithInterval([t-60,t-40], sim.net.pops['ML'].cellGids)
    F_L4 = getFiringRatesWithInterval([t-40,t-20], sim.net.pops['ML'].cellGids)
    F_L5 = getFiringRatesWithInterval([t-20,t], sim.net.pops['ML'].cellGids)
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
    if NBsteps==recordWeightStepSize:
        #if t%recordWeightDT==0:
        print('Weights Recording Time:', t) 
        recordWeights(sim, t)
        NBsteps = 0

def updateInputRates ():
    # update the source firing rates for the R neuron population, based on image contents
    if sim.rank == 0:
        if dconf['verbose'] > 1:
          print(sim.rank,'broadcasting firing rates:',np.where(sim.AIGame.firing_rates==np.amax(sim.AIGame.firing_rates)),np.amax(sim.AIGame.firing_rates))        
        sim.pc.broadcast(sim.AIGame.fvec.from_python(sim.AIGame.firing_rates),0)
        firing_rates = sim.AIGame.firing_rates
    else:
        fvec = h.Vector()
        sim.pc.broadcast(fvec,0)
        firing_rates = fvec.to_python()
        if dconf['verbose'] > 1:
          print(sim.rank,'received firing rates:',np.where(firing_rates==np.amax(firing_rates)),np.amax(firing_rates))                
    # update input firing rates for stimuli to R cells
    lRcell = [c for c in sim.net.cells if c.gid in sim.net.pops['R'].cellGids] # this is the set of R cells
    if dconf['verbose'] > 1: print(sim.rank,'updating len(lRcell)=',len(lRcell),'source firing rates. len(firing_rates)=',len(firing_rates))
    for cell in lRcell:  
        for stim in cell.stims:
            if stim['source'] == 'stimMod':
                stim['hObj'].interval = 1000.0/firing_rates[int(cell.gid)] # interval in ms as a function of rate; is cell.gid correct index???
          
def trainAgent (t):
    """ training interface between simulation and game environment
    """
    global NBsteps, epCount, InputImages
    vec = h.Vector()
    if t<100.0: # for the first time interval use randomly selected actions
        actions =[]
        for _ in range(5):
            action = dconf['movecodes'][random.randint(0,len(dconf['movecodes'])-1)]
            actions.append(action)
    else: #the actions should be based on the activity of motor cortex (MO) 1085-1093
        F_R1 = getFiringRatesWithInterval([t-100,t-80], sim.net.pops['MR'].cellGids) 
        sim.pc.allreduce(vec.from_python([F_R1]), 1) # sum
        F_R1 = vec.to_python()[0] 
        F_R2 = getFiringRatesWithInterval([t-80,t-60], sim.net.pops['MR'].cellGids)
        sim.pc.allreduce(vec.from_python([F_R2]), 1) # sum
        F_R2 = vec.to_python()[0] 
        F_R3 = getFiringRatesWithInterval([t-60,t-40], sim.net.pops['MR'].cellGids)
        sim.pc.allreduce(vec.from_python([F_R3]), 1) # sum
        F_R3 = vec.to_python()[0] 
        F_R4 = getFiringRatesWithInterval([t-40,t-20], sim.net.pops['MR'].cellGids)
        sim.pc.allreduce(vec.from_python([F_R4]), 1) # sum
        F_R4 = vec.to_python()[0] 
        F_R5 = getFiringRatesWithInterval([t-20,t], sim.net.pops['MR'].cellGids)
        sim.pc.allreduce(vec.from_python([F_R5]), 1) # sum
        F_R5 = vec.to_python()[0] 
        F_L1 = getFiringRatesWithInterval([t-100,t-80], sim.net.pops['ML'].cellGids) 
        sim.pc.allreduce(vec.from_python([F_L1]), 1) # sum
        F_L1 = vec.to_python()[0] 
        F_L2 = getFiringRatesWithInterval([t-80,t-60], sim.net.pops['ML'].cellGids)
        sim.pc.allreduce(vec.from_python([F_L2]), 1) # sum
        F_L2 = vec.to_python()[0] 
        F_L3 = getFiringRatesWithInterval([t-60,t-40], sim.net.pops['ML'].cellGids)
        sim.pc.allreduce(vec.from_python([F_L3]), 1) # sum
        F_L3 = vec.to_python()[0] 
        F_L4 = getFiringRatesWithInterval([t-40,t-20], sim.net.pops['ML'].cellGids)
        sim.pc.allreduce(vec.from_python([F_L4]), 1) # sum
        F_L4 = vec.to_python()[0] 
        F_L5 = getFiringRatesWithInterval([t-20,t], sim.net.pops['ML'].cellGids)
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
        rewards, epCount, InputImages = sim.AIGame.playGame(actions, epCount, InputImages)

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
          critic = sum(rewards) # get critic signal (-1, 0 or 1)
        if dconf['verbose']:
          if critic > 0:
            print('REWARD, critic=',critic)
          elif critic < 0:
            critic = -0.01 # reduce magnitude of critic so rewards dominate
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
        for reward in rewards: # this generates an error - since rewards only declared for sim.rank==0; bug?
            sim.allRewards.append(reward)

        for ltpnt in [t-80, t-60, t-40, t-20, t-0]: sim.allTimes.append(ltpnt)

    updateInputRates() # update firing rate of inputs to R population (based on image content)
                
    NBsteps = NBsteps+1
    if NBsteps==recordWeightStepSize:
        #if t%recordWeightDT==0:
        if dconf['verbose'] > 1: print('Weights Recording Time:', t)
        recordAdjustableWeights(sim, t) 
        #recordWeights(sim, t)
        NBsteps = 0

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
sim.analysis.plotData()

if sim.plotWeights: plotWeights() 

if sim.saveWeights:
    #saveWeights(sim, recordWeightDCells)
    saveGameBehavior(sim)
    fid5 = open('data/'+dconf['sim']['name']+'ActionsPerEpisode.txt','w')
    for i in range(len(epCount)):
        fid5.write('\t%0.1f' % epCount[i])
        fid5.write('\n')

InputImages = np.array(InputImages)
print(InputImages.shape)

if sim.saveInputImages:
    with open('data/'+dconf['sim']['name']+'InputImages.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(InputImages.shape))
        for Input_Image in InputImages:
            np.savetxt(outfile, Input_Image, fmt='%-7.2f')
            outfile.write('# New slice\n')
