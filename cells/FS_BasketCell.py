from neuron import h
from math import pi, exp
import numpy as np
###############################################################################
# Soma-targeting interneuron (fast-spiking Basket Cell -- Bas)
###############################################################################
class Bas:
  "Basket cell"	
  def __init__ (self):
    self.soma = soma = h.Section(name='soma',cell=self)
    self.soma.insert('k_ion')
    self.soma.insert('na_ion')
    self.soma.ek = -90 # K+ current reversal potential (mV)
    self.soma.ena = 60 # Na+ current reversal potential (mV)
    self.soma.Ra=100
    self.set_morphology()
    self.set_conductances()

  def set_morphology(self):
    total_area = 10000 # um2
    self.soma.nseg  = 1
    self.soma.cm    = 1      # uF/cm2
    self.soma.diam = np.sqrt(total_area) # um
    self.soma.L    = self.soma.diam/pi  # um			
			
  def set_conductances(self):
    self.soma.insert('pas')
    self.soma.e_pas = -65     # mV
    self.soma.g_pas = 0.1e-3  # S/cm2 
    self.soma.insert('Nafbwb')
    self.soma.insert('Kdrbwb')
	   
#  def set_synapses(self):
#    self.somaGABAf=Synapse(sect=self.soma,loc=0.5,tau1=0.07,tau2=9.1,e=-80);
#    self.somaGABAss=Synapse(sect=self.soma,loc=0.5,tau1=20,tau2=40,e=-80);
#    self.somaAMPA=Synapse(sect=self.soma,loc=0.5,tau1=0.05,tau2=5.3,e=0);
#    self.somaNMDA=SynapseNMDA(sect=self.soma,loc=0.5, tau1NMDA=tau1NMDAEI,tau2NMDA=tau2NMDAEI,r=1,e=0);
		
