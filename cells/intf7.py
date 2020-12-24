from neuron import h
from conf import dconf

# synaptic indices used in intf7.mod NET_RECEIVE
dsyn = {'AM':0, 'NM':1, 'GA':2, 'AM2':3, 'NM2':4, 'GA2':5}
dsyn['AMPA'] = dsyn['AM']
dsyn['NMDA'] = dsyn['NM']
dsyn['GABA'] = dsyn['GA']

class INTF7E ():
  # parameters for excitatory neurons
  dparam = dconf['cell']['E']
  def __init__ (self):
    cell = self.intf = h.INTF7()

class INTF7I ():
  # parameters for fast-spiking interneurons
  dparam = dconf['cell']['I']  
  def __init__ (self):
    cell = self.intf = h.INTF7()

class INTF7IL ():
  # parameters for low-threshold firing interneurons
  dparam = dconf['cell']['IL']  
  def __init__ (self):
    cell = self.intf = h.INTF7()
    
def insertSpikes (sim, dt, spkht=50):
  # inserts spikes into voltage traces (paste-on); depends on NetPyNE simulation data format
  import pandas as pd
  import numpy as np
  sampr = 1e3 / dt # sampling rate
  spkt, spkid = sim.simData['spkt'], sim.simData['spkid']
  spk = pd.DataFrame(np.array([spkid, spkt]).T,columns=['spkid','spkt'])
  for kvolt in sim.simData['V_soma'].keys():
    cellID = int(kvolt.split('_')[1])
    spkts = spk[spk.spkid == cellID]
    if len(spkts):
      for idx in spkts.index:
        tdx = int(spk.at[idx, 'spkt'] * sampr / 1e3)
        sim.simData['V_soma'][kvolt][tdx] = spkht
