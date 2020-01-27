from netpyne import specs, sim
from aigame import SMARTAgent
from neuron import h
import numpy
import random

sim.allTimes = []
sim.allRewards = [] # list to store all rewards
sim.allActions = [] # list to store all actions
sim.allMotorOutputs = [] # list to store firing rate of output motor neurons.
sim.ActionsRewardsfilename = 'ActionsRewards.txt'
sim.MotorOutputsfilename = 'MotorOutputs.txt'
sim.WeightsRecordingTimes = [] 
sim.allRLWeights = [] # list to store weights
sim.allNonRLWeights = [] # list to store weights
sim.RLweightsfilename = 'RLweights.txt'  # file to store weights
sim.NonRLweightsfilename = 'NonRLweights.txt'  # file to store weights
sim.plotWeights = 0  # plot weights
sim.saveWeights = 1  # save weights
sim.saveInputImages = 1 #save Input Images (5 game frames)
recordWeightStepSize = 1
#recordWeightDT = 1000 # interval for recording synaptic weights (change later)
recordWeightDCells = 1 # to record weights for sub samples of neurons

global fid4

fid4 = open(sim.MotorOutputsfilename,'w')

NB_Rneurons = 400
NB_V1neurons = 400
NB_V4neurons = 100
NB_ITneurons = 25

NB_IV1neurons = 100
NB_IV4neurons = 25
NB_IITneurons = 9

NB_MIneurons = 25
NB_MOneurons = 9

NB_IMIneurons = 9

# Network parameters
netParams = specs.NetParams() #object of class NetParams to store the network parameters

#Population parameters
netParams.popParams['R'] = {'cellType': 'E', 'numCells': NB_Rneurons, 'cellModel': 'HH'}  #6400 neurons to represent 6400 pixels, now we have 400 pixels
netParams.popParams['V1'] = {'cellType': 'EV1', 'numCells': NB_V1neurons, 'cellModel': 'HH'} #6400 neurons
netParams.popParams['V4'] = {'cellType': 'EV4', 'numCells': NB_V4neurons, 'cellModel': 'HH'} #1600 neurons
netParams.popParams['IT'] = {'cellType': 'EIT', 'numCells': NB_ITneurons, 'cellModel': 'HH'} #400 neurons

netParams.popParams['IV1'] = {'cellType': 'InV1', 'numCells': NB_IV1neurons, 'cellModel': 'HH'} #1600
netParams.popParams['IV4'] = {'cellType': 'InV4', 'numCells': NB_IV4neurons, 'cellModel': 'HH'} #400
netParams.popParams['IIT'] = {'cellType': 'InIT', 'numCells': NB_IITneurons, 'cellModel': 'HH'} #100

netParams.popParams['MI'] = {'cellType': 'EMI', 'numCells': NB_MIneurons, 'cellModel': 'HH'} #400
netParams.popParams['MO'] = {'cellType': 'EMO', 'numCells': NB_MOneurons, 'cellModel': 'HH'} #100

netParams.popParams['IMI'] = {'cellType': 'InMI', 'numCells': NB_IMIneurons, 'cellModel': 'HH'} #100

netParams.cellParams['ERule'] = {               # cell rule label
        'conds': {'cellType': ['E','EV1','EV4','EIT', 'EMI', 'EMO']},              #properties will be applied to cells that match these conditions
        'secs': {'soma':                        #sections
                {'geom': {'diam':10, 'L':10, 'Ra':120},         #geometry
                'mechs': {'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}}}    #mechanism

netParams.cellParams['IRule'] = {               # cell rule label
        'conds': {'cellType': ['InV1','InV4','InIT', 'InMI']},              #properties will be applied to cells that match these conditions
        'secs': {'soma':                        #sections
                {'geom': {'diam':10, 'L':10, 'Ra':120},         #geometry
                'mechs': {'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}}}    #mechanism

## Synaptic mechanism parameters
netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}  # excitatory synaptic mechanism
netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80}  # inhibitory synaptic mechanism

#netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3, 'e': 0} # AMPA
#netParams.synMechParams['NMDA'] = {'mod': 'Exp2Syn', 'tau1': 0.15, 'tau2': 1.50, 'e': 0} # NMDA
#netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80} # GABAA

STDPparams = {'hebbwt': 0.0001, 'antiwt':-0.00001, 'wmax': 50, 'RLon': 0 , 'RLhebbwt': 0.001, 'RLantiwt': -0.000,
        'tauhebb': 10, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 0, 'verbose':0}

STDPparamsRL = {'hebbwt': 0.00001, 'antiwt':-0.0000, 'wmax': 50, 'RLon': 1 , 'RLhebbwt': 0.00001, 'RLantiwt': -0.000,
        'tauhebb': 10, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 0, 'verbose':0}

netParams.stimSourceParams['stimMod'] = {'type': 'NetStim', 'rate': 'variable', 'noise': 0}
netParams.stimTargetParams['stimMod->all'] = {'source': 'stimMod',
        'conds': {'pop': 'R'},
        'convergence': 1,
        'weight': 0.01,
        'delay': 1,
        'synMech': 'AMPA'}

#background input to inhibitory neurons to increase their firing rate

# Stimulation parameters

netParams.stimSourceParams['ebkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 0.3}
netParams.stimTargetParams['ebkg->all'] = {'source': 'ebkg', 'conds': {'cellType': ['EV1','EV4','EIT', 'EMI', 'EMO']}, 'weight': 0.01, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'}


netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 20, 'noise': 0.3}
netParams.stimTargetParams['bkg->all'] = {'source': 'bkg', 'conds': {'cellType': ['InV1','InV4','InIT', 'InMI']}, 'weight': 0.01, 'delay': 'max(1, normal(5,2))', 'synMech': 'AMPA'}
######################################################################################
def connectLayerswithOverlap(NBpreN, NBpostN, overlap_xdir):
    #NBpreN = 6400 	#number of presynaptic neurons
    NBpreN_x = int(numpy.sqrt(NBpreN))
    NBpreN_y = int(numpy.sqrt(NBpreN))
    #NBpostN = 6400	#number of postsynaptic neurons
    NBpostN_x = int(numpy.sqrt(NBpostN))
    NBpostN_y = int(numpy.sqrt(NBpostN))
    convergence_factor = NBpreN/NBpostN
    convergence_factor_x = numpy.ceil(numpy.sqrt(convergence_factor))
    convergence_factor_y = numpy.ceil(numpy.sqrt(convergence_factor))
    #overlap_xdir = 5	#number of rows in a window for overlapping connectivity
    #overlap_ydir = 5	#number of columns in a window for overlapping connectivity
    overlap_ydir = overlap_xdir
    preNIndices = numpy.zeros((NBpreN_x,NBpreN_y))
    postNIndices = numpy.zeros((NBpostN_x,NBpostN_y))		#list created for indices from linear (1-6400) to square indexing (1-80,81-160,....) 
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
            preN_ind = numpy.where(preNIndices==preN)
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
    NBpreN_x = int(numpy.sqrt(NBpreN))
    NBpreN_y = int(numpy.sqrt(NBpreN))
    NBpostN_x = int(numpy.sqrt(NBpostN))
    NBpostN_y = int(numpy.sqrt(NBpostN))
    divergence_factor = NBpostN/NBpreN
    divergence_factor_x = numpy.ceil(numpy.sqrt(divergence_factor))
    divergence_factor_y = numpy.ceil(numpy.sqrt(divergence_factor))
    overlap_ydir = overlap_xdir
    preNIndices = numpy.zeros((NBpreN_x,NBpreN_y))
    postNIndices = numpy.zeros((NBpostN_x,NBpostN_y))		#list created for indices from linear (1-6400) to square indexing (1-80,81-160,....) 
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
            postN_ind = numpy.where(postNIndices==postN)
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
blistEtoV1 = connectLayerswithOverlap(NBpreN = NB_Rneurons, NBpostN = NB_V1neurons, overlap_xdir = 5)
blistV1toV4 = connectLayerswithOverlap(NBpreN = NB_V1neurons, NBpostN = NB_V4neurons, overlap_xdir = 5)
blistV4toIT = connectLayerswithOverlap(NBpreN = NB_V4neurons, NBpostN = NB_ITneurons, overlap_xdir = 7) #was 15
blistITtoMI = connectLayerswithOverlap(NBpreN = NB_ITneurons, NBpostN = NB_MIneurons, overlap_xdir = 5) #Not sure if this is a good strategy instead of all to all
blistMItoMO = connectLayerswithOverlap(NBpreN = NB_MIneurons, NBpostN = NB_MOneurons, overlap_xdir = 9) #was 19
#blistMItoMO: Feedforward for MI to MO is all to all and can be specified in the connection statement iteself

print('E to V1')
print(blistEtoV1)

print('V1 to V4')
print(blistV1toV4)

print('V4 to IT')
print(blistV4toIT)

print('IT to MI')
print(blistITtoMI)

print('MI to MO')
print(blistMItoMO)


#E to I - Feedforward connections
blistEtoInV1 = connectLayerswithOverlap(NBpreN = NB_Rneurons, NBpostN = NB_IV1neurons, overlap_xdir = 5)
blistV1toInV4 = connectLayerswithOverlap(NBpreN = NB_V1neurons, NBpostN = NB_IV4neurons, overlap_xdir = 7) #was 15
blistV4toInIT = connectLayerswithOverlap(NBpreN = NB_V4neurons, NBpostN = NB_IITneurons, overlap_xdir = 7) #was 15
blistITtoInMI = connectLayerswithOverlap(NBpreN = NB_ITneurons, NBpostN = NB_IMIneurons, overlap_xdir = 7) #was 15

#E to I - WithinLayer connections
blistV1toInV1 = connectLayerswithOverlap(NBpreN = NB_V1neurons, NBpostN = NB_IV1neurons, overlap_xdir = 3)
blistV4toInV4 = connectLayerswithOverlap(NBpreN = NB_V4neurons, NBpostN = NB_IV4neurons, overlap_xdir = 3)
blistITtoInIT = connectLayerswithOverlap(NBpreN = NB_ITneurons, NBpostN = NB_IITneurons, overlap_xdir = 3)
blistMItoInMI = connectLayerswithOverlap(NBpreN = NB_MIneurons, NBpostN = NB_IMIneurons, overlap_xdir = 3)

print('V1 to InV1')
print(blistV1toInV1)

print('V4 to InV4')
print(blistV4toInV4)

print('IT to InIT')
print(blistITtoInIT)

print('MI to InMI')
print(blistMItoInMI)


#I to E - WithinLayer Inhibition
blistInV1toV1 = connectLayerswithOverlapDiv(NBpreN = NB_IV1neurons, NBpostN = NB_V1neurons, overlap_xdir = 5)
blistInV4toV4 = connectLayerswithOverlapDiv(NBpreN = NB_IV4neurons, NBpostN = NB_V4neurons, overlap_xdir = 5)
blistInITtoIT = connectLayerswithOverlapDiv(NBpreN = NB_IITneurons, NBpostN = NB_ITneurons, overlap_xdir = 5)
blistInMItoMI = connectLayerswithOverlapDiv(NBpreN = NB_IMIneurons, NBpostN = NB_MIneurons, overlap_xdir = 5)

print('InV1 to V1')
print(blistInV1toV1)

print('InV4 to V4')
print(blistInV4toV4)

print('InIT to IT')
print(blistInITtoIT)

print('InMI to MI')
print(blistInMItoMI)


#Feedbackward excitation
#E to E  
blistV1toE = connectLayerswithOverlapDiv(NBpreN = NB_V1neurons, NBpostN = NB_Rneurons, overlap_xdir = 3)
blistV4toV1 = connectLayerswithOverlapDiv(NBpreN = NB_V4neurons, NBpostN = NB_V1neurons, overlap_xdir = 3)
blistITtoV4 = connectLayerswithOverlapDiv(NBpreN = NB_ITneurons, NBpostN = NB_V4neurons, overlap_xdir = 3)
blistMItoIT = connectLayerswithOverlapDiv(NBpreN = NB_MIneurons, NBpostN = NB_ITneurons, overlap_xdir = 3)
blistMOtoMI = connectLayerswithOverlapDiv(NBpreN = NB_MOneurons, NBpostN = NB_MIneurons, overlap_xdir = 3)

#Feedforward inhibition
#I to I
blistInV1toInV4 = connectLayerswithOverlap(NBpreN = NB_IV1neurons, NBpostN = NB_IV4neurons, overlap_xdir = 5)
blistInV4toInIT = connectLayerswithOverlap(NBpreN = NB_IV4neurons, NBpostN = NB_IITneurons, overlap_xdir = 5)
blistInITtoInMI = connectLayerswithOverlap(NBpreN = NB_IITneurons, NBpostN = NB_IMIneurons, overlap_xdir = 5)

#Feedbackward inhibition
#I to E 
blistInV1toE = connectLayerswithOverlapDiv(NBpreN = NB_IV1neurons, NBpostN = NB_Rneurons, overlap_xdir = 5)
blistInV4toV1 = connectLayerswithOverlapDiv(NBpreN = NB_IV4neurons, NBpostN = NB_V1neurons, overlap_xdir = 5)
blistInITtoV4 = connectLayerswithOverlapDiv(NBpreN = NB_IITneurons, NBpostN = NB_V4neurons, overlap_xdir = 5)
blistInMItoIT = connectLayerswithOverlapDiv(NBpreN = NB_IMIneurons, NBpostN = NB_ITneurons, overlap_xdir = 5)

#blist = connectRtoV1withOverlap()
#blist = connectRtoV1withoutOverlap()


#Local excitation
#E to E
netParams.connParams['R->R'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'R'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['V1->V1'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'V1'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['V4->V4'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'V4'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['IT->IT'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'IT'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['MI->MI'] = {
        'preConds': {'pop': 'MI'},
        'postConds': {'pop': 'MI'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA'}
#E to I
netParams.connParams['V1->IV1'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'IV1'},
        'connList': blistV1toInV1,
        #'probability': 0.23,
        #'convergence': 9,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['V4->IV4'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'IV4'},
        'connList': blistV4toInV4,
        #'probability': 0.23,
        #'convergence': 9,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['IT->IIT'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'IIT'},
        'connList': blistITtoInIT,
        #'probability': 0.23,
        #'convergence': 9,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['MI->IMI'] = {
        'preConds': {'pop': 'MI'},
        'postConds': {'pop': 'IMI'},
        'connList': blistMItoInMI,
        #'probability': 0.23,
        #'convergence': 9,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
#Local inhibition
#I to E
netParams.connParams['IV1->V1'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'V1'},
        'connList': blistInV1toV1,
        #'probability': 0.02,
        #'divergence': 9,
        'weight': 0.001,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IV4->V4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'V4'},
        'connList': blistInV4toV4,
        #'probability': 0.02,
        #'divergence': 9,
        'weight': 0.001,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IIT->IT'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'IT'},
        'connList': blistInITtoIT,
        #'probability': 0.02,
        #'divergence': 9,
        'weight': 0.001,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IMI->MI'] = {
        'preConds': {'pop': 'IMI'},
        'postConds': {'pop': 'MI'},
        'connList': blistInMItoMI,
        #'probability': 0.02,
        #'divergence': 9,
        'weight': 0.001,
        'delay': 20,
        'synMech': 'GABA'}
#I to I
netParams.connParams['IV1->IV1'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV1'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IV4->IV4'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IV4'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IIT->IIT'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'IIT'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IMI->IMI'] = {
        'preConds': {'pop': 'IMI'},
        'postConds': {'pop': 'IMI'},
        'probability': 0.02,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'GABA'}
#E to E feedforward connections
netParams.connParams['R->V1'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'V1'},
        'connList': blistEtoV1,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['V1->V4'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'V4'},
        'connList': blistV1toV4,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['V4->IT'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'IT'},
        'connList': blistV4toIT,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['IT->MI'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'MI'},
        'connList': blistITtoMI,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}
netParams.connParams['MI->MO'] = {
        'preConds': {'pop': 'MI'},
        'postConds': {'pop': 'MO'},
        'connList': blistMItoMO,
        #'convergence': 100,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}
#E to I feedforward connections
netParams.connParams['R->IV1'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'IV1'},
        'connList': blistEtoInV1,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['V1->IV4'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistV1toInV4,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['V4->IIT'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'IIT'},
        'connList': blistV4toInIT,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['IT->IMI'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'IMI'},
        'connList': blistITtoInMI,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}

#E to E feedbackward connections
netParams.connParams['V1->R'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'R'},
        'connList': blistV1toE,
        #'convergence': 10,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['V4->V1'] = {
        'preConds': {'pop': 'V4'},
        'postConds': {'pop': 'V1'},
        'connList': blistInV4toV1,
        #'convergence': 10,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['IT->V4'] = {
        'preConds': {'pop': 'IT'},
        'postConds': {'pop': 'V4'},
        'connList': blistITtoV4,
        #'convergence': 10,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA'}
netParams.connParams['MI->IT'] = {
        'preConds': {'pop': 'MI'},
        'postConds': {'pop': 'IT'},
        'connList': blistMItoIT,
        #'convergence': 10,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}
netParams.connParams['MO->MI'] = {
        'preConds': {'pop': 'MO'},
        'postConds': {'pop': 'MI'},
        'connList': blistMOtoMI,
        #'convergence': 10,
        'weight': 0.0001,
        'delay': 20,
        'synMech': 'AMPA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}

#I to E connections

netParams.connParams['IV1->R'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'R'},
        'connList': blistInV1toE,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IV4->V1'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'V1'},
        'connList': blistInV4toV1,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IIT->V4'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'V4'},
        'connList': blistInITtoV4,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IMI->IT'] = {
        'preConds': {'pop': 'IMI'},
        'postConds': {'pop': 'IT'},
        'connList': blistInMItoIT,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'GABA'}
#I to I
netParams.connParams['IV1->IV4'] = {
        'preConds': {'pop': 'IV1'},
        'postConds': {'pop': 'IV4'},
        'connList': blistInV1toInV4,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IV4->IIT'] = {
        'preConds': {'pop': 'IV4'},
        'postConds': {'pop': 'IIT'},
        'connList': blistInV4toInIT,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'GABA'}
netParams.connParams['IIT->IMI'] = {
        'preConds': {'pop': 'IIT'},
        'postConds': {'pop': 'IMI'},
        'connList': blistInITtoInMI,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'GABA',
        'plast': {'mech': 'STDP', 'params': STDPparamsRL}}
#Simulation options
simConfig = specs.SimConfig()           # object of class SimConfig to store simulation configuration

simConfig.duration = 1e4                      # Duration of the simulation, in ms
simConfig.dt = 0.2                            # Internal integration timestep to use
simConfig.verbose = False                       # Show detailed messages
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordCellsSpikes = [-1]
simConfig.recordStep = 0.2                      # Step size in ms to save data (e.g. V traces, LFP, etc)
simConfig.filename = 'model_output'  # Set file output name
simConfig.savePickle = False            # Save params, network and sim output to pickle file
simConfig.saveMat = True

#simConfig.analysis['plotRaster'] = True                         # Plot a raster
simConfig.analysis['plotTraces'] = {'include': [1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092]}
#simConfig.analysis['plotRaster'] = {'timeRange': [500,1000],'popRates':'overlay','saveData':'RasterData.pkl','showFig':True}
simConfig.analysis['plotRaster'] = {'popRates':'overlay','saveData':'RasterData.pkl','showFig':True}
#simConfig.analysis['plot2Dnet'] = True 
#simConfig.analysis['plotConn'] = True           # plot connectivity matrix
###################################################################################################################################

#SMARTAgent.initGame('self')

sim.SMARTAgent = SMARTAgent()


def recordWeights (sim, t):
    """ record the STDP weights during the simulation - called in trainAgent
    """
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
    print('Firing rate : %.3f Hz'%(avgRates))
    return avgRates

InputImages = []
NBsteps = 0
epCount = []
last_obs = [] #make sure this does not introduce a bug
def trainAgentFake(t):
    """ training interface between simulation and game environment
    """
    global NBsteps, last_obs, epCount, InputImages
    if t<21.0: # for the first time interval use first action randomly and other four actions based on relative position of ball and agent.
        last_obs = []
        rewards, actions, last_obs, epCount, InputImages = sim.SMARTAgent.playGameFake(last_obs, epCount, InputImages)
    else: #the actions are generated based on relative positions of ball and Agent.
        rewards, actions, last_obs, epCount, InputImages = sim.SMARTAgent.playGameFake(last_obs, epCount, InputImages)
    print('actions generated by model are: ', actions)
    F_R1 = getFiringRatesWithInterval([t-20,t], [1085])
    F_R2 = getFiringRatesWithInterval([t-20,t], [1086])
    F_R3 = getFiringRatesWithInterval([t-20,t], [1087])
    F_R4 = getFiringRatesWithInterval([t-20,t], [1088])
    F_R5 = getFiringRatesWithInterval([t-20,t], [1088,1089])
    F_L1 = getFiringRatesWithInterval([t-20,t], [1089,1090])
    F_L2 = getFiringRatesWithInterval([t-20,t], [1090])
    F_L3 = getFiringRatesWithInterval([t-20,t], [1091])
    F_L4 = getFiringRatesWithInterval([t-20,t], [1092])
    F_L5 = getFiringRatesWithInterval([t-20,t], [1093])
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
        print('t=',t,'- adjusting weights based on RL critic value:', critic)
        for cell in sim.net.cells:
            for conn in cell.conns:
                STDPmech = conn.get('hSTDP')  # check if has STDP mechanism
                if STDPmech:   # run stdp.mod method to update syn weights based on RLprint cell.gid
                    STDPmech.reward_punish(float(critic))
    print('rewards are : ', rewards)
    for action in actions:
        sim.allActions.append(action)
    for reward in rewards:
        sim.allRewards.append(reward)
    ltpnt = t-20
    for _ in range(5):
        ltpnt = ltpnt+4
        sim.allTimes.append(ltpnt)
    sim.SMARTAgent.run(t,sim)
    print('trainAgent time is : ', t)
    NBsteps = NBsteps+1
    if NBsteps==recordWeightStepSize:
        #if t%recordWeightDT==0:
        print('Weights Recording Time:', t) 
        recordWeights(sim, t)
        NBsteps = 0

def trainAgent(t):
    """ training interface between simulation and game environment
    """
    global NBsteps, InputImages
    if t<21.0: # for the first time interval use randomly selected actions
        actions =[]
        for _ in range(5):
            action = random.randint(3,4)
            actions.append(action)
    else: #the actions should be based on the activity of motor cortex (MO) 1085-1093
        F_R1 = getFiringRatesWithInterval([t-20,t], [1085])
        F_R2 = getFiringRatesWithInterval([t-20,t], [1086])
        F_R3 = getFiringRatesWithInterval([t-20,t], [1087])
        F_R4 = getFiringRatesWithInterval([t-20,t], [1088])
        F_R5 = getFiringRatesWithInterval([t-20,t], [1088,1089])
        F_L1 = getFiringRatesWithInterval([t-20,t], [1089,1090])
        F_L2 = getFiringRatesWithInterval([t-20,t], [1090])
        F_L3 = getFiringRatesWithInterval([t-20,t], [1091])
        F_L4 = getFiringRatesWithInterval([t-20,t], [1092])
        F_L5 = getFiringRatesWithInterval([t-20,t], [1093])
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
            actions.append(4) #UP
        elif F_R1<F_L1:
            actions.append(3) # Down
        else:
            actions.append(random.randint(3,4))
            #actions.append(1) # No move 
        if F_R2>F_L2:
            actions.append(4) #UP
        elif F_R2<F_L2:
            actions.append(3) #Down
        else:
            actions.append(random.randint(3,4))
            #actions.append(1) #No move
        if F_R3>F_L3:
            actions.append(4) #UP
        elif F_R3<F_L3:
            actions.append(3) #Down
        else:
            actions.append(random.randint(3,4))
            #actions.append(1) #No move
        if F_R4>F_L4:
            actions.append(4) #UP
        elif F_R4<F_L4:
            actions.append(3) #Down
        else:
            actions.append(random.randint(3,4))
            #actions.append(1) #No move
        if F_R5>F_L5:
            actions.append(4) #UP
        elif F_R5<F_L5:
            actions.append(3) #Down
        else:
            actions.append(random.randint(3,4))
            #actions.append(1) #No move
    print('actions generated by model are: ', actions)
    rewards, epCount, InputImages = sim.SMARTAgent.playGame(actions, epCount, InputImages)
    #I don't understand the code below. Copied from Salva's RL model
    vec = h.Vector()
    if sim.rank == 0:
        rewards, epCount, InputImages = sim.SMARTAgent.playGame(actions, epCount, InputImages)
        critic = sum(rewards) # get critic signal (-1, 0 or 1)
        sim.pc.broadcast(vec.from_python([critic]), 0) # convert python list to hoc vector for broadcast data received from arm
    else: # other workers
        sim.pc.broadcast(vec, 0)
        critic = vec.to_python()[0] #till here I dont understand
    if critic != 0: # if critic signal indicates punishment (-1) or reward (+1)
        print('t=',t,'- adjusting weights based on RL critic value:', critic)
        for cell in sim.net.cells:
            for conn in cell.conns:
                STDPmech = conn.get('hSTDP')  # check if has STDP mechanism
                if STDPmech:   # run stdp.mod method to update syn weights based on RLprint cell.gid
                    STDPmech.reward_punish(float(critic))
    print('rewards are : ', rewards)
    for action in actions:
        sim.allActions.append(action)
    for reward in rewards:
        sim.allRewards.append(reward)
    ltpnt = t-20
    for _ in range(5):
        ltpnt = ltpnt+4
        sim.allTimes.append(ltpnt)
    sim.SMARTAgent.run(t,sim)
    print('trainAgent time is : ', t)
    NBsteps = NBsteps+1
    if NBsteps==recordWeightStepSize:
        #if t%recordWeightDT==0:
        print('Weights Recording Time:', t) 
        recordWeights(sim, t)
        NBsteps = 0

#Alterate to create network and run simulation
sim.initialize(                       # create network object and set cfg and net params
    simConfig = simConfig,   # pass simulation config and network params as arguments
    netParams = netParams)
sim.net.createPops()                      # instantiate network populations
sim.net.createCells()                     # instantiate network cells based on defined populations
sim.net.connectCells()                    # create connections between cells based on params
sim.net.addStims()                      #instantiate netStim
sim.setupRecording()                  # setup variables to record for each cell (spikes, V traces, etc)
#sim.runSim()
sim.runSimWithIntervalFunc(100.0,trainAgentFake)
sim.gatherData()
sim.saveData()
sim.analysis.plotData()

if sim.plotWeights: 
    plotWeights() 
if sim.saveWeights:
    saveWeights(sim, recordWeightDCells)
    saveGameBehavior(sim)
    fid5 = open('ActionsPerEpisode.txt','w')
    for i in range(len(epCount)):
        fid5.write('\t%0.1f' % epCount[i])
        fid5.write('\n')

InputImages = numpy.array(InputImages)
print(InputImages.shape)

if sim.saveInputImages:
    with open('InputImages.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(InputImages.shape))
        for Input_Image in InputImages:
            numpy.savetxt(outfile, Input_Image, fmt='%-7.2f')
            outfile.write('# New slice\n')
