from netpyne import specs, sim
from aigame import SMARTAgent
from neuron import h
import numpy

# Network parameters
netParams = specs.NetParams() #object of class NetParams to store the network parameters

#Population parameters
netParams.popParams['R'] = {'cellType': 'E', 'numCells': 6400, 'cellModel': 'HH'}
netParams.popParams['V1'] = {'cellType': 'EV1', 'numCells': 400, 'cellModel': 'HH'}

netParams.cellParams['ERule'] = {               # cell rule label
        'conds': {'cellType': ['E','EV1']},              #properties will be applied to cells that match these conditions
        'secs': {'soma':                        #sections
                {'geom': {'diam':10, 'L':10, 'Ra':120},         #geometry
                'mechs': {'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}}}    #mechanism

## Synaptic mechanism parameters
netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0}  # excitatory synaptic mechanism

STDPparams = {'hebbwt': 0.0001, 'antiwt':-0.00001, 'wmax': 50, 'RLon': 0 , 'RLhebbwt': 0.001, 'RLantiwt': -0.000,
        'tauhebb': 10, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 0, 'verbose':0}

netParams.stimSourceParams['stimMod'] = {'type': 'NetStim', 'rate': 'variable', 'noise': 0}
netParams.stimTargetParams['stimMod->all'] = {'source': 'stimMod',
        'conds': {'pop': 'R'},
        'convergence': 1,
        'weight': 0.01,
        'delay': 1,
        'synMech': 'exc'}

######################################################################################

NBpreN = 6400
NBpostN = 400
convergence_factor = NBpreN/NBpostN
xdir = int(numpy.sqrt(convergence_factor))
ydir = int(numpy.sqrt(convergence_factor))
blist = []
for ix in range(xdir):
    blist.append([ix,0])
nlist = blist
for iy in range(ydir-1):
    nlist = numpy.ndarray.tolist(numpy.add(nlist,[int(numpy.sqrt(NBpreN)),0]))
    blist = blist + nlist
#blist = [[0,0],[1,0],[2,0],[3,0],[80,0],[81,0],[82,0],[83,0],[160,0],[161,0],[162,0],[163,0],[240,0],[241,0],[242,0],[243,0]]
acol = numpy.ndarray.tolist(numpy.add(blist,[xdir*int(numpy.sqrt(NBpreN)),int(numpy.sqrt(NBpostN))]))
arow = numpy.ndarray.tolist(numpy.add(blist,[xdir,1]))
#b.append(a)
for rowNB in range(int(numpy.sqrt(NBpostN))):
    for colNB in range(int(numpy.sqrt(NBpostN))-1):
        blist = blist + arow
        arow = numpy.ndarray.tolist(numpy.add(arow,[xdir,1]))
    if rowNB<19:
        blist = blist + acol
        arow = acol
        arow = numpy.ndarray.tolist(numpy.add(arow,[xdir,1]))
        acol = numpy.ndarray.tolist(numpy.add(acol,[xdir*int(numpy.sqrt(NBpreN)),int(numpy.sqrt(NBpostN))]))

######################################################################################


netParams.connParams['R->V1'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'V1'},
        'connList': blist,
        #'convergence': 10,
        'weight': 0.001,
        'delay': 20,
        'synMech': 'exc',
        'plast': {'mech': 'STDP', 'params': STDPparams}}

#Simulation options
simConfig = specs.SimConfig()           # object of class SimConfig to store simulation configuration

simConfig.duration = 1*1e4                      # Duration of the simulation, in ms
simConfig.dt = 0.2                            # Internal integration timestep to use
simConfig.verbose = False                       # Show detailed messages
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordStep = 0.2                      # Step size in ms to save data (e.g. V traces, LFP, etc)
simConfig.filename = 'model_output'  # Set file output name
simConfig.savePickle = False            # Save params, network and sim output to pickle file

simConfig.analysis['plotRaster'] = True                         # Plot a raster
simConfig.analysis['plotTraces'] = {'include': [4, 164, 324, 484, 6401]}
###################################################################################################################################

#SMARTAgent.initGame('self')

sim.SMARTAgent = SMARTAgent()

def trainAgent(t):
    sim.SMARTAgent.playGame()
    sim.SMARTAgent.run(t,sim)


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
sim.runSimWithIntervalFunc(20.0,trainAgent)
sim.gatherData()
sim.saveData()
sim.analysis.plotData()

