from netpyne import specs, sim
from arm import Arm
from neuron import h
# Network parameters
netParams = specs.NetParams() #object of class NetParams to store the network parameters

#Population parameters
netParams.popParams['S'] = {'cellType': 'E', 'numCells': 4, 'cellModel': 'HH'}
netParams.popParams['S2'] = {'cellType': 'E2', 'numCells': 4, 'cellModel': 'HH'}

netParams.cellParams['ERule'] = {		# cell rule label
	'conds': {'cellType': ['E','E2']}, 		#properties will be applied to cells that match these conditions
	'secs': {'soma':			#sections
		{'geom': {'diam':10, 'L':10, 'Ra':120},		#geometry
		'mechs': {'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}}}	#mechanism


## Synaptic mechanism parameters
netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0}  # excitatory synaptic mechanism


netParams.connParams['S->S2'] = { #  S -> S2 label
        'preConds': {'pop': 'S'}, # conditions of presyn cells
        'postConds': {'pop': 'S2'}, # conditions of postsyn cells
        'probability': 1,             # probability of connection
        'weight': 0.01,                         # synaptic weight
        'delay': 5,                                     # transmission delay (ms)
        'synMech': 'exc'}               # synaptic mechanism

#Stimulation parameters
#Need to make firing rate variable and driven by some other population

netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 100, 'noise': 0.3}
netParams.stimTargetParams['bkg->all'] = {'source': 'bkg',
	'conds': {'pop': 'S'},
	'weight': 0.01,
	'delay': 1,
	'synMech': 'exc'}


netParams.stimSourceParams['stimMod'] = {'type': 'NetStim', 'rate': 'variable', 'noise': 0}
netParams.stimTargetParams['stimMod->all'] = {'source': 'stimMod',
	'conds': {'pop': 'S2'},
	'weight': 0.01,
	'delay': 1,
	'synMech': 'exc'}

#Simulation options
simConfig = specs.SimConfig()           # object of class SimConfig to store simulation configuration

simConfig.duration = 1*1e3                      # Duration of the simulation, in ms
simConfig.dt = 0.025                            # Internal integration timestep to use
simConfig.verbose = False                       # Show detailed messages
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordStep = 0.1                      # Step size in ms to save data (e.g. V traces, LFP, etc)
simConfig.filename = 'model_output'  # Set file output name
simConfig.savePickle = False            # Save params, network and sim output to pickle file

simConfig.analysis['plotRaster'] = True                         # Plot a raster
simConfig.analysis['plotTraces'] = {'include': [0, 1, 2, 5, 6, 7]}                     # Plot recorded traces for this list of cells
#simConfig.analysis['plot2Dnet'] = True           # plot 2D visualization of cell positions and connections


# Create network and run simulation
#sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

def runArm(t):
    Arm.run(t, sim)

#def modUnit(t, sim):
#    sim.net.cells[4].stims[0][‘hObj'].interval 
#= sim.net.cells[4].stims[0][‘hObj'].interval/10 
#  sim.net.cells[5].stims[0][‘hObj'].interval = sim.net.cells[4].stims[0][‘hObj'].interval/5
#  sim.net.cells[6].stims[0][‘hObj'].interval = sim.net.cells[4].stims[0][‘hObj'].interval/2
#  sim.net.cells[7].stims[0][‘hObj'].interval = sim.net.cells[4].stims[0][‘hObj'].interval/20

#Alterate to create network and run simulation
sim.initialize(                       # create network object and set cfg and net params
    simConfig = simConfig,   # pass simulation config and network params as arguments
    netParams = netParams)
sim.net.createPops()                      # instantiate network populations
sim.net.createCells()                     # instantiate network cells based on defined populations
sim.net.connectCells()                    # create connections between cells based on params
sim.net.addStims()			#instantiate netStim
sim.setupRecording()                  # setup variables to record for each cell (spikes, V traces, etc)
#sim.runSim()
sim.runSimWithIntervalFunc(1.0,runArm)
sim.gatherData()
sim.saveData()
sim.analysis.plotData()
