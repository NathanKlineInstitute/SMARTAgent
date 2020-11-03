from neuron import h
dummy = h.Section()

class IntFire4Cell (): 
  '''Create an IntFire4 cell based on 2007 parameterization using either izhi2007.mod (no hosting section) or izhi2007b.mod (v in created section)
  If host is omitted or None, this will be a section-based version that uses Izhi2007b with state vars v, u where v is the section voltage
  If host is given then this will be a shared unused section that simply houses an Izhi2007 using state vars V and u
'''
  def __init__ (self):
    self.type=type
    self.sec = dummy
    self.intf = h.IntFire4(0.5, sec=self.sec) # Create a new u,V 2007 neuron at location 0.5 (doesn't matter where)    
  #def init (self): self.sec(0.5).v = self.vinit
