COMMENT
intf7.mod was branched/simplified from intf6.mod on 20nov2 by samn

artificial cell incorporating 4 input weights with different time constants and signs
typically a fast AMPA, slow NMDA, and fast GABAA
features:
  1. Mg dependence for NMDA activation
  2. depolarization blockade
  3. AHP affects both Vm and refractory period  (adaptation)
  4. decrementing excitatory and/or inhibitory activity post spk (another adaptation)
note that artificial cells only do calculations when they receive events
ENDCOMMENT

:* NEURON, PARAMETER, ASSIGNED blocks
NEURON {
  ARTIFICIAL_CELL INTF7
  RANGE VAM, VNM, VGA, AHP           :::: cell state variables
  RANGE VAM2, VNM2, VGA2                  :::: state vars for distal dend inputs
  RANGE Vm                                :::: derived var
  : parameters
  RANGE tauAM, tauNM, tauGA            :::: synaptic params
  RANGE tauAM2, tauNM2, tauGA2         :::: synaptic params meant for distal dends
  RANGE tauahp, ahpwt                  :::: intrinsic params
  RANGE tauRR , RRWght                 :::: relative refrac. period tau, wght of Vblock-VTH for refrac
  RANGE RMP,VTH,Vblock,VTHC,VTHR       :::: Vblock for depol blockade
  RANGE incRR : whether allow VTHC to increment past RRWght*(Vblock-VTH) over successive refrac periods
  RANGE refrac,AHP2REF                  :::: refrac period and extender
  RANGE mg0                             :::: sensitivity to Mg2+, used in rates
  RANGE maxnmc                          :::: maximum NMDA 'conductance', used in rates
  GLOBAL EAM, ENM, EGA,mg               :::: "reverse potential" distance from rest
  GLOBAL spkht                          :::: display: spike height
  : other stuff
  RANGE  t0,tg,twg,refractory,trrs :::: t0,tg save times for analytic calc
  GLOBAL RES      :::: table look up values for exp
}

: PARAMETER block - sets all variables to defaults at start
PARAMETER {
  tauAM = 10 (ms)
  tauNM = 300 (ms)
  tauGA = 10 (ms)
  tauAM2 = 20 (ms)
  tauNM2 = 300 (ms)
  tauGA2 = 20 (ms)
  ahpwt = 0
  tauahp= 10 (ms)
  tauRR = 6 (ms)
  refrac = 5 (ms)
  AHP2REF = 0.0 : default is no refrac period increment/decrmenet
  RRWght = 0.75
  VTH = -45      : fixed spike threshold
  VTHC = -45
  VTHR = -45
  incRR = 0
  Vblock = -20   : level of depolarization blockade
  mg = 1.6       : for NMDA Mg dep.
  RMP=-65
  EAM = 65 : deviation above RMP
  ENM = 90
  EGA = -15
  spkht = 50 : height of paste-on spike (handled on python side)
  STDAM=0
  STDNM=0
  STDGA=0
  mg0 = 3.57
  maxnmc = 1.0
}

ASSIGNED {
  Vm VAM VNM VGA AHP VAM2 VNM2 VGA2
  t0 tg twg refractory trrs
  RES
}

:* CONSTRUCTOR, DESTRUCTOR, INITIAL
:** CONSTRUCT: create a structure to save the identity of this unit and char integer flags
CONSTRUCTOR {
}

DESTRUCTOR {
}

:** INITIAL
INITIAL { LOCAL id
  reset() 
  t0 = 0
  tg = 0
  twg = 0
  trrs = 0
}

PROCEDURE reset () {
  Vm = RMP
  VAM = 0
  VNM = 0
  VGA = 0
  AHP=0
  VAM2 = 0
  VNM2 = 0
  VGA2 = 0
  t0 = t
  tg = t
  twg = t
  trrs = t
  refractory = 0 : 1 means cell is absolute refractory
  VTHC=VTH :set current threshold to absolute threshold value
  VTHR=VTH :set this one too to make sure it is initialized
}

:* NET_RECEIVE
NET_RECEIVE (wAM,wNM,wGA,wAM2,wNM2,wGA2,wflg) { LOCAL tmp,id
  INITIAL { wAM=wAM wNM=wNM wGA=wGA wAM2=wAM2 wNM2=wNM2 wGA2=wGA2 wflg=0}
  :printf("DB0: flag=%g Vm=%g",flag,VAM+VNM+VGA+RMP+AHP+VAM2+VNM2+VGA2)
  :if (flag==0) { printf(" (%g %g %g %g %g %g %g)",wAM,wNM,wGA,wAM2,wNM2,wGA2,wflg) }
  :printf("\n")
  : causes of spiking: between VTH and Vblock
  : update state variables: VAM, VNM, VGA
  if (VAM>hoc_epsilon)  { VAM = VAM*EXP(-(t - t0)/tauAM) } else { VAM=0 } :AMPA
  if (VNM>hoc_epsilon)  { VNM = VNM*EXP(-(t - t0)/tauNM) } else { VNM=0 } :NMDA
  if (VGA< -hoc_epsilon){ VGA = VGA*EXP(-(t - t0)/tauGA) } else { VGA=0 } :GABAA    
  if (VAM2>hoc_epsilon) {VAM2 = VAM2*EXP(-(t - t0)/tauAM2) } else { VAM2=0 } :AMPA from distal dends
  if (VNM2>hoc_epsilon) {VNM2 = VNM2*EXP(-(t - t0)/tauNM2) } else { VNM2=0 } :NMDA from distal dends
  if (VGA2< -hoc_epsilon){VGA2 = VGA2*EXP(-(t - t0)/tauGA2) } else { VGA2=0 } :GABAA more distal from soma   

  if(refractory==0){:once refractory period over, VTHC falls back towards VTH
    if(VTHC>VTH) { :eg, for decelerating cells after firing, thresh increases
      VTHC = VTH + (VTHR-VTH)*EXP(-(t-trrs)/tauRR) 
    } else if(RRWght<0 && VTHC<VTH) { :eg, for accelerating cells after firing, thresh decreases
      VTHC = VTH - (VTHR-VTH)*EXP(-(t-trrs)/tauRR) 
    }
  }
  if (AHP< -hoc_epsilon){ AHP = AHP*EXP(-(t-t0)/tauahp) } else { AHP=0 } : adaptation
  t0 = t : finished using t0
  Vm = VAM+VNM+VGA+AHP+VAM2+VNM2+VGA2 : membrane deviation from rest
  if (Vm> -RMP) {Vm= -RMP}: 65 mV above rest
  if (Vm<  RMP) {Vm= RMP} : 65 mV below rest
  :*** only add weights if an external excitation
  if (flag==0) { 
    : AMPA Erev=0 (0-RMP==65 mV above rest)
    if (wAM>0) {
      VAM = VAM + wAM*(1-Vm/EAM)
      if (VAM>EAM) { 
      } else if (VAM<0) { VAM=0 }
    }
    if (wAM2>0) { : AMPA from distal dends
      VAM2 = VAM2 + wAM2*(1-Vm/EAM)
      if (VAM2>EAM) { 
      } else if (VAM2<0) { VAM2=0 }
    }
    : NMDA; Mg effect based on total activation in rates()
    if (wNM>0 && VNM<ENM) { 
      VNM = VNM + wNM*rates(RMP+Vm)*(1-Vm/ENM) 
      if (VNM>ENM) { 
      } else if (VNM<0) { VNM=0 }
    }
    if (wNM2>0 && VNM2<ENM) { : NMDA from distal dends
      VNM2 = VNM2 + wNM2*rates(RMP+Vm)*(1-Vm/ENM)
      if (VNM2>ENM) { 
      } else if (VNM2<0) { VNM2=0 }
    }
    : GABAA , GABAA2 : note that all wts are positive
    if (wGA>0 && VGA>EGA) { : the neg here gives the inhibition
      VGA = VGA - wGA*(1-Vm/EGA) 
      if (VGA<EGA) { 
      } else if (VGA>0) { VGA=0 } : if want reversal of VGA need to also edit above
    }
    if (wGA2>0 && VGA2>EGA) { : the neg here gives the inhibition, GABAA2, inputs further from soma
      VGA2 = VGA2 - wGA2*(1-Vm/EGA)
      if (VGA2<EGA) { 
      } else if (VGA2>0) { VGA2=0 } : if want reversal of VGA2 need to also edit above
    }
  } else if (flag==3) { 
    refractory = 0 :end of absolute refractory period    
    trrs = t : save time of start of relative refractory period
VERBATIM
    return; //@ done
ENDVERBATIM
  }
  :** check for Vm>VTH -> fire
  Vm = VAM+VNM+VGA+RMP+AHP+VAM2+VNM2+VGA2 : WARNING -- Vm defined differently than above
  if (Vm>0)   {Vm= 0 }
  if (Vm<-90) {Vm=-90}  
  if (refractory==0 && Vm>VTHC) {
VERBATIM
    if (Vm>Vblock) {//@ do nothing
      return; 
    }
ENDVERBATIM
    AHP = AHP - ahpwt
    tmp=t
    net_event(tmp)
    if(incRR) { : additive
      VTHC=VTHC+RRWght*(Vblock-VTH):increase threshold for relative refrac. period. NB: RRWght can be < 0
      if(VTHC > Vblock) {VTHC=Vblock} else if(VTHC < RMP) {VTHC=RMP}
    } else { : non-additive
      VTHC=VTH+RRWght*(Vblock-VTH):increase threshold for relative refrac. period. NB: RRWght can be < 0
    }
    VTHR=VTHC :starting thresh value for relative refrac period, keep track of it
    refractory = 1 : abs. refrac on = do not allow any more spikes/bursts to begin (even for IB cells)
    net_send(refrac-AHP*AHP2REF, 3) :event for end of abs. refrac., sent separately for IB cells @ end of burst
  }
}

:* TABLES
PROCEDURE EXPo (x) {
  TABLE RES FROM -20 TO 0 WITH 5000
  RES = exp(x)
}

FUNCTION EXP (x) {
  EXPo(x)
  EXP = RES
}


FUNCTION rates (vv) {
  : from Stevens & Jahr 1990a,b
  rates = maxnmc / (1 + exp(0.062 (/mV) * -vv) * ( (mg / mg0) ) )
}

