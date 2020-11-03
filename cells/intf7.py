from neuron import h

# synaptic indices used in intf7.mod NET_RECEIVE
dsyn = {'AM':0, 'NM':1, 'GA':2, 'AM2':3, 'NM2':4, 'GA2':5}
dsyn['AMPA'] = dsyn['AM']
dsyn['NMDA'] = dsyn['NM']
dsyn['GABA'] = dsyn['GA']

class INTF7E ():
  # parameters for excitatory neurons
  dparam = {"ahpwt":1,
            "tauahp":400,
            "RMP": -65,
            "VTH": -40,
            "refrac":  5,
            "Vblock": -25,
            "tauGA": 10,
            "tauGA2": 20,
            "tauAM2": 20,
            "tauNM2": 300,
            "tauRR": 1,
            "RRWght": 0.25}  
  def __init__ (self):
    cell = self.intf = h.INTF7()
    cell.ahpwt=1
    cell.tauahp=400
    cell.RMP= -65
    cell.VTH= -40 
    cell.refrac=  5
    cell.Vblock= -25    
    cell.tauGA  = 10
    cell.tauGA2 = 20
    cell.tauAM2 = 20
    cell.tauNM2 = 300            
    cell.tauRR = 1
    cell.RRWght = .25

class INTF7I ():
  # parameters for fast-spiking interneurons
  dparam = {"ahpwt":0.5,
            "tauahp":50,
            "RMP": -63,
            "VTH": -40,
            "refrac":  2.5,
            "Vblock": -10,
            "tauGA": 10,
            "tauGA2": 20,
            "tauAM2": 20,
            "tauNM2": 300,            
            "tauRR": 1,
            "RRWght": 0.25}    
  def __init__ (self):
    cell = self.intf = h.INTF7()
    cell.ahpwt=0.5
    cell.refrac= 2.5
    cell.tauahp=50
    cell.Vblock=-10    
    cell.RMP = -63
    cell.VTH= -40
    cell.tauGA  = 10
    cell.tauGA2 = 20
    cell.tauAM2 = 20
    cell.tauNM2 = 300    
    cell.tauRR = 1
    cell.RRWght = 0.25
    
def insertSpikes (sim, spkht=50):
  # inserts spikes into voltage traces (paste-on); depends on NetPyNE simulation data format
  import pandas as pd
  import numpy as np
  sampr = 1e3 / simConfig.dt # sampling rate
  spkt, spkid = sim.simData['spkt'], sim.simData['spkid']
  spk = pd.DataFrame(np.array([spkid, spkt]).T,columns=['spkid','spkt'])
  for kvolt in sim.simData['Vsoma'].keys():
    cellID = int(kvolt.split('_')[1])
    spkts = spk[spk.spkid == cellID]
    if len(spkts):
      for idx in spkts.index:
        tdx = int(spk.at[idx, 'spkt'] * sampr / 1e3)
        sim.simData['Vsoma'][kvolt][tdx] = spkht
