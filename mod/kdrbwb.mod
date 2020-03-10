UNITS {
  (mA) = (milliamp)
  (mV) = (millivolt)
  (mS) = (millisiemens)
}

NEURON {
  SUFFIX Kdrbwb
  USEION k WRITE ik
  RANGE phin,gkdr,ek
  RANGE taon,ninf
}
	
PARAMETER {
  gkdr =   9 (mS/cm2)
  ek   = -90 (mV)
  phin = 5
}
    
ASSIGNED {
  v       (mV)
  ik      (mA/cm2)
  celsius (degC)
  ninf    (1)
  taon    (ms)
}

STATE { n }

PROCEDURE iassign () { ik = (1e-3) * gkdr * n^4 * (v-ek) }

INITIAL { 
  rates(v)
  n  = ninf
  iassign()
}

BREAKPOINT {
  SOLVE states METHOD cnexp	
  iassign()
}

DERIVATIVE states { 
  rates(v)
  n' = (ninf-n)/taon
}

PROCEDURE rates(v(mV)) { LOCAL an, bn, q10
  q10  = phin:^((celsius-27.0(degC))/10.0(degC))
    
  an = fun3(v,  -34,  -0.01,   -10)
  bn = fun1(v,  -44,   0.125,  -80)
    
  ninf = an/(an+bn)
  taon = 1./((an+bn)*q10)
}

INCLUDE "aux_fun.inc"
