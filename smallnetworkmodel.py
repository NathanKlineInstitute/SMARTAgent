from neuron import h
from netpyne import specs, sim
import pylab; pylab.show()  # this line is only necessary in certain systems where figures appear empty

# Network parameters
netParams = specs.NetParams()  # object of class NetParams to store the network parameters

## Population parameters
netParams.popParams['pop1'] = {'cellType': 'PYR', 'numCells': 3, 'cellModel': 'HH'}
netParams.popParams['pop2'] = {'cellType': 'PYR', 'numCells': 3, 'cellModel': 'HH'}

## Cell property rules
cellRule = {'conds': {'cellType': 'PYR'},  'secs': {}} 	# cell rule dict
cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}  														# soma params dict
cellRule['secs']['soma']['geom'] = {'diam': 18.8, 'L': 18.8, 'Ra': 123.0}  									# soma geometry
cellRule['secs']['soma']['mechs']['hh'] = {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}  		# soma hh mechanism
netParams.cellParams['PYRrule'] = cellRule  												# add dict to list of cell params

## Synaptic mechanism parameters
netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0}  # excitatory synaptic mechanism

# Stimulation parameters
netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 10, 'noise': 0.5}
netParams.stimTargetParams['bkg->PYR'] = {'source': 'bkg', 'conds': {'cellType': 'PYR'}, 'weight': 0.01, 'delay': 5, 'synMech': 'exc'}

## Cell connectivity rules

netParams.connParams['pop1->pop2'] = {  
  'preConds': {'pop': 'pop1'},  # conditions of presyn cells
  'postConds': {'pop': 'pop2'}, # conditions of postsyn cells
  'divergence': 12,       # probability of connection
  'weight': 0.01,         # synaptic weight
  'delay': 2,           # transmission delay (ms)
  'synMech': 'exc'  }       # synaptic mechanism
k = 'pop1->pop2'

netParams.connParams[k]['plast'] = {'mech': 'STDP','params': {'wbase':0.0000001,'wmax':0.00048,'RLon':1,'RLlenhebb':1600,'RLlenanti':50,'useRLexp':1,'RLhebbwt':0.000004,'RLantiwt':0.0,'hebbwt':0,'antiwt':0,'tauhebb':10,'RLwindhebb':50,'softthresh':0,'verbose':0}}

# Simulation options
simConfig = specs.SimConfig()		# object of class SimConfig to store simulation configuration

simConfig.duration = 1*1e3 			# Duration of the simulation, in ms
simConfig.dt = 0.025 				# Internal integration timestep to use
simConfig.verbose = False  			# Show detailed messages
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordStep = 0.1 			# Step size in ms to save data (eg. V traces, LFP, etc)
simConfig.filename = 'model_output'  # Set file output name
simConfig.savePickle = False 		# Save params, network and sim output to pickle file
simConfig.saveJson = True 	

simConfig.analysis['plotRaster'] = True 			# Plot a raster
simConfig.analysis['plotTraces'] = {'include': [1]} 			# Plot recorded traces for this list of cells
simConfig.analysis['plot2Dnet'] = True           # plot 2D visualization of cell positions and connections

# Create network and run simulation
sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

sim.analysis.plotRaster(showFig=1, saveFig=1)
sim.analysis.plotTraces(showFig=1, saveFig=1)
sim.analysis.plot2Dnet(showFig=1, saveFig=1)
simConfig.analysis['plotRaster'] = {'saveFig': True}            # Plot a raster
simConfig.analysis['plotTraces'] = {'include': [1], 'saveFig': True}            # Plot recorded traces for this list of cells
simConfig.analysis['plot2Dnet'] = {'saveFig': True}            # plot 2D visualization of cell positions and connections


