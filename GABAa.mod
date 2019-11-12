NEURON {  POINT_PROCESS GABAa }
: copied from "Synaptic information transfer in computer models of neocortical columns (Neymotin et al. 2010)"
PARAMETER {
  Cdur	= 1.08	(ms)		: transmitter duration (rising phase)
  Alpha	= 1.	(/ms mM)	: forward (binding) rate
  Beta	= 0.5	(/ms)		: backward (unbinding) rate
  Erev	= -70	(mV)		: reversal potential
}

INCLUDE "netcon.inc"
