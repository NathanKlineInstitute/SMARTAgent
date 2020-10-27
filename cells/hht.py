from neuron import h

class HHE:
    def __init__ (self):
        self.soma = soma = h.Section(name='soma',cell=self)
        soma.diam=soma.L=18.8
        soma.Ra=123
        soma.insert('hh')
        soma(.5).hh.gnabar=.12
        soma(.5).hh.gkbar=0.036
        soma(.5).hh.gl=0.003
        soma(.5).hh.el=-70

class HHI:
    def __init__ (self):
        self.soma = soma = h.Section(name='soma',cell=self)
        soma.diam=soma.L=18.8
        soma.Ra=123
        soma.insert('hh')        
        soma(.5).hh.gnabar=.12
        soma(.5).hh.gkbar=0.036
        soma(.5).hh.gl=0.003
        soma(.5).hh.el=-70

        
