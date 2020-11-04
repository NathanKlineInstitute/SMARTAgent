"""
tut_artif.py 
Tutorial on artificial cells (no sections)
"""
from netpyne import specs, sim
from netpyne.specs import Dict
netParams = specs.NetParams()  # object of class NetParams to store the network parameters
simConfig = specs.SimConfig()  # dictionary to store sets of simulation configurations
simConfig.hParams['celsius'] = 37

###############################################################################
# NETWORK PARAMETERS
###############################################################################
# Population parameters
numCells = 100
connList = [[i,i] for i in range(numCells)]

netParams.popParams['artif1'] = {'cellModel': 'NetStim', 'numCells': numCells, 'rate':  'variable', 'noise': 0, 'start': 0, 'seed': 2}  # pop of NetStims

useArtif = True # False

netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}  # excitatory synaptic mechanism

if useArtif:
  netParams.popParams['artif3'] = {'cellModel': 'INTF7', 'numCells': numCells}#, 'taue': 5.0, 'taui1':10,'taui2':20,'taum':50}  # pop of IntFire4
  simConfig.recordTraces = {'V_soma':{'var':'Vm'}}  # Dict with traces to record  
else:
  netParams.popParams['artif3'] = {'numCells': numCells, 'cellModel': 'Mainen'}
  #netParams.importCellParams(label='PYR_Mainen_rule', conds={'cellType': ['artif3']}, fileName='cells/mainen.py', cellName='PYR2')
  netParams.importCellParams(label='PYR_Mainen_rule', conds={'cellType': ['artif3']}, fileName='cells/mainen.py', cellName='PYR2')
  netParams.cellParams['PYR_Mainen_rule']['secs']['soma']['threshold'] = 0.0
  simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record

# Connections
k = 'artif1->artif3'
netParams.connParams[k] = {
    'preConds': {'pop': 'artif1'}, 'postConds': {'pop': 'artif3'},
    #'probability': 0.2,
    'connList': connList,
    'weight': 10,
    #'synMech': 'AMPA',                
    'delay': 'uniform(1,5)',
    'weightIndex': 0}

# netParams.connParams[k]['plast'] = {'mech': 'STDP', 'params': {'RLon':1,'RLlenhebb':200,'RLhebbwt':0.001,'RLwindhebb':50,'wbase':0,'wmax':2}}

lsynweights = []

def recordAdjustableWeights (sim, t, popname='artif3'):
  # record the plastic weights for specified popname
  lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[popname].cellGids] # this is the set of MR cells
  for cell in lcell:
    for conn in cell.conns:
      if 'hSTDP' in conn:
        lsynweights.append([t,conn.preGid,cell.gid,float(conn['hObj'].weight[0])])
  return len(lcell)

###############################################################################
# SIMULATION PARAMETERS
###############################################################################
# Simulation parameters
simConfig.duration = 1*1e3 # Duration of the simulation, in ms
simConfig.dt = 0.1 # Internal integration timestep to use
simConfig.createNEURONObj = 1  # create HOC objects when instantiating network
simConfig.createPyStruct = 1  # create Python structure (simulator-independent) when instantiating network
simConfig.verbose = 1 #False  # show detailed messages 
# Recording 
# # Analysis and plotting 
simConfig.analysis['plotRaster'] = True
#simConfig.analysis['plotTraces'] = {'include': ['all']}
simConfig.analysis['plotTraces'] = {'include': [('artif3',0)]}

sim.create(netParams, simConfig)

lcell = [c for c in sim.net.cells if c.gid in sim.net.pops['artif1'].cellGids]
for cell in lcell: cell.hPointp.interval = 5

"""
lSTDPmech = []
for cell in sim.net.cells:
  for conn in cell.conns:
    STDPmech = conn.get('hSTDP')  # check if the connection has a NEURON STDP mechanism object
    if STDPmech: lSTDPmech.append(STDPmech)
"""

###############################################################################
# RUN SIM
###############################################################################

def mycallback (t):
  print('mycallback',t)
  # for stdpmech in lSTDPmech: stdpmech.reward_punish(1.0)
  # recordAdjustableWeights(sim,t)

def insertSpikes (sim, spkht=50):
    sampr = 1e3 / simConfig.dt
    import pandas as pd
    import numpy as np
    spkt, spkid = sim.simData['spkt'], sim.simData['spkid']
    spk = pd.DataFrame(np.array([spkid, spkt]).T,columns=['spkid','spkt'])
    for kvolt in sim.simData['V_soma'].keys():
        cellID = int(kvolt.split('_')[1])
        spkts = spk[spk.spkid == cellID]
        if len(spkts):
            for idx in spkts.index:
                tdx = int(spk.at[idx, 'spkt'] * sampr / 1e3)
                sim.simData['V_soma'][kvolt][tdx] = spkht

lcell3 = [c for c in sim.net.cells if c.gid in sim.net.pops['artif3'].cellGids]
c = lcell3[0]
  
sim.run.runSim()
#sim.runSimWithIntervalFunc(50,mycallback)

insertSpikes(sim)

sim.gatherData()

sim.analysis.plotData()

from pylab import *
ion()
figure()
plot(sim.simData['t'],sim.simData['V_soma']['cell_100'])

"""
#sim.createSimulateAnalyze()
sim.create(netParams, simConfig) 
sim.gatherData() # gather data from different nodes
sim.analysis.plotData()    

"""
