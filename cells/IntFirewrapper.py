from neuron import h
dummy = h.Section()

class IntFire4Cell (): 
  '''Create an izhikevich cell based on 2007 parameterization using either izhi2007.mod (no hosting section) or izhi2007b.mod (v in created section)
  If host is omitted or None, this will be a section-based version that uses Izhi2007b with state vars v, u where v is the section voltage
  If host is given then this will be a shared unused section that simply houses an Izhi2007 using state vars V and u
  Note: Capacitance 'C' differs from sec.cm which will be 1; vr is RMP; vt is threshold; vpeak is peak voltage
'''

  def __init__ (self, type='RS', host=None, cellid=-1):
    self.type=type
    if host is None:  # need to set up a sec for this
      self.sec=h.Section(name='IntFire'+type+str(cellid))
      self.sec.L, self.sec.diam, self.sec.cm = 10, 10, 31.831 # empirically tuned
      self.intf = h.IntFire4(0.5, sec=self.sec) 
      self.vinit = -60
    else: 
      self.sec = dummy
      self.intf = h.IntFire4(0.5, sec=self.sec) # Create a new u,V 2007 neuron at location 0.5 (doesn't matter where) 

    #self.izh.C,self.izh.k,self.izh.vr,self.izh.vt,self.izh.vpeak,self.izh.a,self.izh.b,self.izh.c,self.izh.d,self.izh.celltype = type2007[type]
    #self.izh.cellid   = cellid # Cell ID for keeping track which cell this is
  
  def init (self): self.sec(0.5).v = self.vinit

  """
  def reparam (self, type='RS', cellid=-1):
    self.type=type
    self.izh.C,self.izh.k,self.izh.vr,self.izh.vt,self.izh.vpeak,self.izh.a,self.izh.b,self.izh.c,self.izh.d,self.izh.celltype = type2007[type]
    self.izh.cellid = cellid # Cell ID for keeping track which cell this is
  """
