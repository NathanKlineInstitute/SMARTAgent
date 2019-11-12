from netpyne import specs, sim
from aigame import SMARTAgent
from neuron import h
import numpy

sim.allWeights = [] # list to store weights
sim.weightsfilename = 'weights.txt'  # file to store weights
sim.plotWeights = 0  # plot weights

# Network parameters
netParams = specs.NetParams() #object of class NetParams to store the network parameters

#Population parameters
netParams.popParams['R'] = {'cellType': 'E', 'numCells': 6400, 'cellModel': 'HH'}
netParams.popParams['V1'] = {'cellType': 'EV1', 'numCells': 6400, 'cellModel': 'HH'}
netParams.popParams['V4'] = {'cellType': 'EV4', 'numCells': 1600, 'cellModel': 'HH'}
netParams.popParams['IT'] = {'cellType': 'EIT', 'numCells': 400, 'cellModel': 'HH'}

netParams.cellParams['ERule'] = {               # cell rule label
        'conds': {'cellType': ['E','EV1','EV4','EIT']},              #properties will be applied to cells that match these conditions
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
def connectRtoV1withOverlap(NBpreN, NBpostN, overlap_xdir):
    #NBpreN = 6400 	#number of presynaptic neurons
    NBpreN_x = int(numpy.sqrt(NBpreN))
    NBpreN_y = int(numpy.sqrt(NBpreN))
    #NBpostN = 6400	#number of postsynaptic neurons
    NBpostN_x = int(numpy.sqrt(NBpostN))
    NBpostN_y = int(numpy.sqrt(NBpostN))
    convergence_factor = NBpreN/NBpostN
    convergence_factor_x = numpy.sqrt(convergence_factor)
    convergence_factor_y = numpy.sqrt(convergence_factor)
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
                preN = int(convergence_factor_x*convergence_factor_y*NBpostN_y*i) + int(convergence_factor_y*j)
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
      
def connectRtoV1withoutOverlap():
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
    return blist

#####################################################################################
def saveWeights(sim):
    ''' Save the weights for each plastic synapse '''
    count = 0
    with open(sim.weightsfilename,'w') as fid:
        for weightdata in sim.allWeights:
            count = count+1
            if count==1000:
                #fid.write('%0.0f' % weightdata[0]) # Time
                for i in range(1,len(weightdata)): fid.write('\t%0.8f' % weightdata[i])
                fid.write('\n')
                count = 0
    print(('Saved weights as %s' % sim.weightsfilename))    


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

######################################################################################
blist = connectRtoV1withOverlap(NBpreN = 6400, NBpostN = 6400, overlap_xdir = 5)
blistV1toV4 = connectRtoV1withOverlap(NBpreN = 6400, NBpostN = 1600, overlap_xdir = 5)
#blist = connectRtoV1withOverlap()
#blist = connectRtoV1withoutOverlap()
netParams.connParams['R->V1'] = {
        'preConds': {'pop': 'R'},
        'postConds': {'pop': 'V1'},
        'connList': blist,
        #'convergence': 10,
        'weight': 0.002,
        'delay': 20,
        'synMech': 'exc',
        'plast': {'mech': 'STDP', 'params': STDPparams}}
netParams.connParams['V1->V4'] = {
        'preConds': {'pop': 'V1'},
        'postConds': {'pop': 'V4'},
        'connList': blistV1toV4,
        #'convergence': 10,
        'weight': 0.01,
        'delay': 20,
        'synMech': 'exc',
        'plast': {'mech': 'STDP', 'params': STDPparams}}
#Simulation options
simConfig = specs.SimConfig()           # object of class SimConfig to store simulation configuration

simConfig.duration = 1e4                      # Duration of the simulation, in ms
simConfig.dt = 0.2                            # Internal integration timestep to use
simConfig.verbose = False                       # Show detailed messages
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordStep = 0.2                      # Step size in ms to save data (e.g. V traces, LFP, etc)
simConfig.filename = 'model_output'  # Set file output name
simConfig.savePickle = False            # Save params, network and sim output to pickle file

simConfig.analysis['plotRaster'] = True                         # Plot a raster
simConfig.analysis['plotTraces'] = {'include': [13000, 13500, 14000]}
###################################################################################################################################

#SMARTAgent.initGame('self')

sim.SMARTAgent = SMARTAgent()

def trainAgent(t):
    sim.SMARTAgent.playGame()
    sim.SMARTAgent.run(t,sim)
    sim.allWeights.append([]) # Save this time
    for cell in sim.net.cells:
        for conn in cell.conns:
            if 'hSTDP' in conn:
                sim.allWeights[-1].append(float(conn['hObj'].weight[0])) # save weight only for STDP conns


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

if sim.plotWeights:
    saveWeights(sim) 
    plotWeights() 
