: $Id: intf6.mod,v 1.100 2012/04/05 22:38:25 samn Exp $

:* main COMMENT
COMMENT

intf6.mod was branched from intf.mod version 847 on 10jul13 -- look at intf.mod RCS
log/diffs to see anything prior. note that AM2,NM2,GA2 code was mostly taken from
intf.mod version 815.

artificial cell incorporating 4 input weights with different time constants and signs
typically a fast AMPA, slow NMDA, and fast GABAA
features:
  1. Mg dependence for NMDA activation
  2. depolarization blockade
  3. AHP affects both Vm and refractory period  (adaptation)
  4. decrementing excitatory and/or inhibitory activity post spk (another adaptation)
since artificial cells only do calculations when they receive events, a set of vec
  pointers are maintained to allow state var information storage when event arrives
  (see initrec() and record())
ENDCOMMENT

:* main VERBATIM block
VERBATIM

#include "misc.h"

static int ctt(unsigned int, char**);
static int setdvi2(double*,double*,char*,int,int,double*,double*);
static int setdvi3(double*,double*,char*,int,double*,double*);
void freesywv();
// Definitions for synaptic scaling procs
void raise_activity_sensor(double time);
void decay_activity_sensor(double time);
void update_scale_factor(double time);
void dynamicdelete(double time);
double get_avg_activity();

#define PI 3.14159265358979323846264338327950288419716939937510
#define nil 0
#define CTYPp 100 // CTYPp>CTYPi from labels.hoc
#define SOP (((id0*) _p_sop)->vp)
#define IDP (*((id0**) &(_p_sop)))
#define NSW 100  // just store voltages
#define NSV 12//10 state variables (+1 for time,+1 for extra field at end and extra offset in loops in record)
#define FOFFSET 100 // flag offset for net_receive()
#define WRNUM 5  // a single INTF6 can store into this many ww field vecs
#define DELM(X,Y) (*(pg->delm+(X)*CTYPi+(Y)))
#define DELD(X,Y) (*(pg->deld+(X)*CTYPi+(Y)))
#define DVG(X,Y) ((int)*(pg->dvg+(X)*CTYPi+(Y)))
// #define DVG(X,Y,Z) ((int)*(pg->dvg+(X)*CTYPi+(Y)))
#define WMAT(X,Y,Z) (*(pg->wmat+(X)*CTYPi*STYPi+(Y)*STYPi+(Z)))
#define WD0(X,Y,Z)  (*(pg->wd0 +(X)*CTYPi*STYPi+(Y)*STYPi+(Z)))
#define NUMC(X) (*(pg->numc+(X)))
#define HVAL(X) (*(hoc_objectdata[(hoc_get_symbol((X)))->u.oboff]._pval))
#define HPTR(X) (hoc_objectdata[(hoc_get_symbol((X)))->u.oboff]._pval)

// for recording (?)
typedef struct VPT {
 unsigned int  id;
 unsigned int  size;
 unsigned int  p;
 void*    vv[NSV];
 double* vvo[NSV];
} vpt;

// each column can have one of these
typedef struct POSTGRP { // postsynaptic group
  double *dvg; double *delm; double *deld; double *ix; double *ixe; double *wmat; double *wd0;
  double *numc; // num cells by type
  unsigned int col; // COLUMN ID
  double* jrid; // for recording SPIKES
  double* jrtv;
  void* jridv;
  void* jrtvv;
  unsigned int jtpt,jtmax,jrmax; 
  unsigned long jri,jrj;
  unsigned long spktot,eventtot;
  double *isp, *vsp, *wsp, *sysp; // arrays for external inputs
  int  vspn;
  double *lastspk; // array with last spike times for all cells
  unsigned int cesz; // size of ce
  Object *ce; // cell list
  struct POSTGRP *next;
} postgrp;

// each cell gets one of these, note that postgrp pointer is an element
typedef struct ID0 {
  vpt     *vp;
  postgrp *pg; // <-- pointer to get to postsynaptic cells, shared by cells in a column
  float    wscale[WRNUM];
  Point_process **dvi; // each cell has a divergence list
  Point_process **cvi; // each cell has a convergence list
  double *del;         // each syn has its own intrinsic delay
  char *syns;          // each syn has a type
  unsigned char *sprob;    // each syn has a firing probability 0-255->0-1
  double* wgain; // gain for synapses - used for plasticity
  double* pplasttau; // plasticity tau for synapse
  double* pplastinc; // plasticity inc for synapse (max inc)
  double* pplastmaxw; // max weight gain for plasticity
  double* pdope; // dopamine eligibility

  ////////////////////////////////////////////////////////////////////////////////////////////////
  //     THE PARAMETERS IN THIS 'BLOCK' ARE ASSOCIATED WITH HOMEOSTATIC SYNAPTIC SCAING
  double activity; // Slow-varying cell activity value
  double max_err; // Maximum saturation value for the activity sensor
  double max_scale; // Maximum scaling factor
  double lastupdate; // Time of last activity sensor decay / spike update
  double goal_activity; // Target firing rate  
  double activity_integral_err; // Integral record of cell's activity divergence from target activity
  double scalefactor; // Derived activity-dependent scaling factor, by which to multiply AMPA weights
  ////////////////////////////////////////////////////////////////////////////////////////////////

  int* peconv; // IDs of E cells converging on this cell
  int econvsz; // # of E cells converging on this cell
  int* piconv; // IDs of I cells convering on this cell
  int iconvsz; // # of I cells converging on this cell
  double* syw1; // synaptic weights (parallel to divergence list) -- used for AMPA,GABAA
  double* syw2; // synaptic weights -- used for NMDA,GABAB -- these lists only used when wsetting==1
  unsigned int dvt;
  unsigned int  id; // within-COLUMN ID
  unsigned int col; // COLUMN
  unsigned int rvb;
  unsigned int rvi;
  unsigned int spkcnt;
  unsigned int blkcnt;
  unsigned int gid; // global ID
  int rve;
  char   wreci[WRNUM]; // since use -1 as a flag
  char   errflag;
  // type -> vbr MUST REMAIN unbroked BLOCK -- see flag()
  // when adding flags also augment iflags, iflnum
  // only use first 3 letters with flag() -- see iflags
  unsigned char     type;  // | 
  unsigned char     inhib; // | 
  unsigned char     record;// |
  unsigned char     wrec;  // |
  unsigned char     jttr;  // |
  unsigned char     input; // |
  unsigned char     vinflg;// |
  unsigned char     invl0; // |
  unsigned char     jcn;   // |
  unsigned char     dead;  // |
  unsigned char     vbr;   // |
           char     dbx;   // |
           char     flag;  // |
           char     out;   // |
  // end BLOCK
} id0;

// globals -- range vars must be malloc'ed in the CONSTRUCTOR
static double activityoneovertau; // for homeostatic synaptic scaling: Store 1/tau for faster calculations
static vpt *vp; // vp, pg, ip are used as temporary pointers
static id0 *ip, *qp, *rp;
static int inumcols=0;
static int ippgbufsz=0;
static postgrp **ppg=0x0;
static postgrp *pg;
static Object *CTYP;
static Point_process *pmt, *tpnt;
static char *name;
static Symbol* cbsv;
// iflags string use to find flags -- note that only 1st 3 chars are used to identify
static char iflags[100]="typ inh rec wre jtt inp vin inv jcn dea vbr dbx fla out"; 
static char iflnum=14, iflneg=11, errflag;      // turn on after generating an error message
static double *jsp, *invlp;
static id0 *lop(), *lopr(), *getlp(); // accessed by all INTF6, get pointer from list
static void applyEXSTDP (id0* ppo,double pospkt); // apply standard STDP from E->X cells
static void applyIXSTDP (id0* ppo,double pospkt); // apply STDP from I->X cells
static void applyEDOPE (id0* ppo,double pospkt); // apply DOPAMINE eligibility
static void applyIDOPE (id0* ppo,double pospkt); // apply DOPAMINE eligibility
static double vii[NSV];   // temp storage
static unsigned int wwpt,wwsz,wwaz; // pointer, size for shared ww vectors
static unsigned int sead, spikes[CTYPp], blockcnt[CTYPp]; // 'sead' vs global 'seed'/ used elsewhere
static unsigned int AMo[CTYPp],NMo[CTYPp],GAo[CTYPp]; // count overages for types
static unsigned int AMo2[CTYPp],NMo2[CTYPp],GAo2[CTYPp]; // count overages for types (farther from soma)
static char* CNAME[CTYPp]; // 20 should be > CTYPi
static int cty[CTYPp], process, ctymap[CTYPp];
static int CTYN, CTYPi, STYPi, dscrsz; // from labels.hoc
static double qlimit, *dscr;
FILE *wf1, *wf2, *tf;
void*    ww[NSW];
double* wwo[NSW];
static int AM=0, NM=1, GA=2, GB=3, AM2=4, NM2=5, GA2=6, SU=3, IN=4, DP=2; // from labels.hoc
static double wts[13],hsh[13];  // for jitcons to use as a junk pointer
static int spkoutf2();
ENDVERBATIM

:* NEURON, PARAMETER, ASSIGNED blocks
NEURON {
  ARTIFICIAL_CELL INTF6
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
  RANGE nbur,tbur,refrac,AHP2REF        :::: burst size, interval; refrac period and extender
  RANGE invl,oinvl,WINV,invlt           :::: interval bursting params
  RANGE Vbrefrac                        
  RANGE STDAM, STDNM, STDGA             :::: specific amounts of STD for each type of synapse
                                        :::: NB: before using STDAM,STDNM,STDGA need to debug/check
                                        :::: for possible unintended interations with wts,_args 
                                        :::: to make sure no interference with the weights in net_receive
                                        ::::
  RANGE mg0                             :::: sensitivity to Mg2+, used in rates
  RANGE maxnmc                          :::: maximum NMDA 'conductance', used in rates
  GLOBAL EAM, ENM, EGA,mg               :::: "reverse potential" distance from rest
  GLOBAL spkht, wwwid,wwht              :::: display: spike height, width/ht for pop spikes
  GLOBAL stopoq                         :::: flags: stop if q is empty, use STD
  : other stuff
  POINTER sop                          :::: Structure pointer for other range vars
  RANGE  spck,xloc,yloc,zloc
  RANGE  t0,tg,twg,refractory,trrs :::: t0,tg save times for analytic calc
  RANGE  cbur                         :::: burst statevar
  RANGE  WEX                          :::: weight of external input < 0 == inhib, > 0 ==excit
  RANGE  EXSY                         :::: synapse target of external input
  RANGE  lfpscale                     :::: scales contribution to lfp, only if cell is being recorded in wrecord
  GLOBAL vdt,nxt,RES,ESIN,Psk      :::: table look up values for exp,sin
  GLOBAL prnum, nsw, rebeg             :::: for debugging moves
  GLOBAL subsvint, jrsvn, jrsvd, jrtime, jrtm :::: output params
  GLOBAL DEAD_DIV, seedstep            :::: dead cells on div list?
  GLOBAL seaddvioff                    :::: seed offset for dvi/del 
  GLOBAL WVAR,DELMIN
  GLOBAL savclock,slowset,FLAG  
  GLOBAL tmax,installed,verbose        :::: simplest output
  GLOBAL pathbeg,pathend,PATHMEASURE,pathidtarg,pathtytarg,seadsetting,pathlen
  GLOBAL maxplastt : maximum difference in time between spikes to apply plasticity over
  GLOBAL plaststartT : when plasticity is turned on
  GLOBAL plastendT   : when plasticity is turned off
  GLOBAL resetplast  : whether to reset all wgain entries to 1 at start of run
  GLOBAL wsetting : setting for weights. 0=use WMAT,WD0. 1=use syw1,syw2.
  GLOBAL ESTDP : whether to use STDP @ E->X synapses
  GLOBAL ISTDP : whether to use STDP @ I->X synapses
  GLOBAL SOFTSTDP : whether to use soft bounds for STDP
  GLOBAL EPOTW,EDEPW,IPOTW,IDEPW : STDP potentiation vs depression factors for increments
  GLOBAL nextGID : don't mess with this unless have a good reason!
  GLOBAL EDOPE : whether using dopamine-style learning for E->X weights
  GLOBAL IDOPE : whether using dopamine-style learning for I->X weights
  GLOBAL DOPE : whether using dopamine-style learning
  GLOBAL FORWELIGTR : forward (pre-to-post-synaptic propagation) eligibility traces
  GLOBAL BACKELIGTR : backward (post-to-pre-synaptic propagation) eligibility traces
  GLOBAL EXPELIGTR : use an exponential decay for the eligibility traces?
  GLOBAL maxeligtrdur: maximum eligibilty trace duration (in ms)
  GLOBAL reseteligtr : reset eligibility trace after synapse rewarded/punished  

  : VARIABLES RELATING TO HOMEOSTATIC SYNAPTIC SCALING (IMPLEMENTED BY MARK ROWAN)
  GLOBAL scaling            : Is compensatory scaling switched on for all cells? Default is off. Globally set.
  GLOBAL dynamicdel         : Is dynamic scaling factor-proportional deletion switched on? Default is off.
  GLOBAL delspeed           : Rate constant for spontaneous deletion (Alzheimer's experiments)
  GLOBAL scaleinhib         : Set to TRUE (1) for I-cell scaling in addition to E-cell scaling. Default is off (0).
  GLOBAL activitytau        : Activity time constant (ms^-1)  
  GLOBAL activitybeta       : Scaling strength constant (s^-1 Hz^-1)
  GLOBAL activitygamma      : Scaling update constant (s^-2 Hz^-1)
}

: PARAMETER block - sets all variables to defaults at start
PARAMETER {
  tauAM = 10 (ms)
  tauNM = 300 (ms)
  tauGA = 10 (ms)
  tauAM2 = 20 (ms)
  tauNM2 = 300 (ms)
  tauGA2 = 20 (ms)
  invl =  100 (ms)
  WINV =  0
  ahpwt = 0
  tauahp= 10 (ms)
  tauRR = 6 (ms)
  refrac = 5 (ms)
  AHP2REF = 0.0 : default is no refrac period increment/decrmenet
  Vbrefrac = 20 (ms)
  RRWght = 0.75
  wwwid = 10
  wwht = 10
  VTH = -45      : fixed spike threshold
  VTHC = -45
  VTHR = -45
  incRR = 0
  Vblock = -20   : level of depolarization blockade
  vdt = 0.1      : time step for saving state var
  mg = 1         : for NMDA Mg dep.
  sop=0
  nbur=1
  tbur=2
  RMP=-65
  EAM = 65
  ENM = 90
  EGA = -15
  spkht = 50
  prnum = -1
  nsw=0
  rebeg=0
  subsvint=0
  jrsvn=1e4 jrsvd=1e4 jrtime=-1 jrtm=-1
  seedstep=44340
  seaddvioff=9102098713763e-134
  DEAD_DIV=1
  WVAR=0.2
  stopoq=0
  PATHMEASURE=0
  verbose=1
  seadsetting=0
  pathidtarg=-1
  DELMIN=1e-5 : min delay to bother using queue -- otherwise considered simultaneous
  STDAM=0
  STDNM=0
  STDGA=0
  mg0 = 3.57
  maxnmc = 1.0
  lfpscale = 1.0
  maxplastt = 10.0
  plaststartT = -1 : default of -1 means always on (when seadsetting==3)
  plastendT = -1   : default of -1 means always on (when seadsetting==3)
  resetplast = 1   : default to reset wgain entries to 1 at start of run
  wsetting = 0 : default -- use WMAT,WD0
  ISTDP = 0 : no I->X STDP by default
  ESTDP = 1 : E->X STDP by default, when t in bounds of plaststartT,plastendT and plasticity on (seadsetting==3)
  SOFTSTDP = 1 : by default uses soft-bounds
  EPOTW = 1 : Weight by which STDP produces potentiation if t(post)>t(pre) at an E->[anything] synapse
  EDEPW = 1 : can bias towards depression by having EDEPW > 1 or EPOTW < 1
  IPOTW = 1
  IDEPW = 1
  nextGID = 0
  DOPE = 0 : no dopamine-based learning by default
  EDOPE = 0 : no dopamine-based learning by default
  IDOPE = 0 : no dopamine-based learning by default
  FORWELIGTR = 1 : turn on forward eligibility traces by default
  BACKELIGTR = 0 : turn off backward eligibility traces by default
  EXPELIGTR = 1 : turn on exponential decay of eligibilty traces by default
  maxeligtrdur = 100.0 : set maximum eligibility trace time to 100 ms by default
  reseteligtr = 0 : don't reset by default

  : default values for homeostatic synaptic scaling
  scaling = 0                          : Compensatory synaptic scaling defaults to 'off'
  dynamicdel = 0                       : Dynamic deletion defaults to 'off'
  delspeed = 0.0                       : Rate constant for dynamic deletion
  scaleinhib = 0                       : Whether or not we should scale I cells as well as E cells
  activitytau = 100.0e3                : Activity sensor time constant (ms^-1) (van Rossum et al., 2000)
  activitybeta = 4.0e-8                : was e-5 Scaling strength constant (s^-1 Hz^-1) (van Rossum et al., 2000)
  activitygamma = 1.0e-10              : was e-7 Scaling update constant (s^-2 Hz^-1) (van Rossum et al., 2000)
}

ASSIGNED {
  Vm VAM VNM VGA AHP VAM2 VNM2 VGA2
  t0 tg twg refractory nxt xloc yloc zloc trrs
  WEX EXSY RES ESIN Psk cbur invlt oinvl tmax spck savclock slowset FLAG
  installed
  pathbeg pathend pathtytarg pathlen
}

:* CONSTRUCTOR, DESTRUCTOR, INITIAL
:** CONSTRUCT: create a structure to save the identity of this unit and char integer flags
CONSTRUCTOR {
  VERBATIM 
  { int lid,lty,lin,lco,lgid,i; unsigned int sz;
    if (ifarg(1)) { lid=(int) *getarg(1); } else { lid= UINT_MAX; } // ID
    if (ifarg(2)) { lty=(int) *getarg(2); } else { lty= -1; } // type
    if (ifarg(3)) { lin=(int) *getarg(3); } else { lin= -1; } // inhib
    if (ifarg(4)) { lco=(int) *getarg(4); } else { lco= -1; } // column
    _p_sop = (void*)ecalloc(1, sizeof(id0)); // important that calloc sets all flags etc to 0
    ip = IDP;
    ip->id=lid; ip->type=lty; ip->inhib=lin; ip->col=lco; 
    ip->pg=0x0; ip->dvi=0x0; ip->del=0x0; ip->sprob=0x0; 
    ip->syns=0x0; ip->wgain=0x0; ip->peconv=ip->piconv=0x0; ip->syw1=ip->syw2=0x0;
    ip->pplasttau=0x0; ip->pplastinc=0x0; ip->pplastmaxw=0x0; ip->pdope=0x0;
    ip->dead = ip->invl0 = ip->record = ip->jttr = ip->input = 0; // all flags off
    ip->dvt = ip->vbr = ip->wrec = ip->jcn = ip->out = 0;
    for (i=0;i<WRNUM;i++) {ip->wreci[i]=-1; ip->wscale[i]=-1.0;}
    ip->rve=-1;
    pathbeg=-1;
    slowset=0;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //     THE PARAMETERS IN THIS 'BLOCK' ARE ASSOCIATED WITH HOMEOSTATIC SYNAPTIC SCAING
    ip->activity = 0; // Sensor for this cell's recent activity (default 0MHz i.e. cycles per ms)
    ip->max_err = 0; // Max error value
    ip->max_scale = 100; // Max scaling factor
    ip->lastupdate = 0; // Time of last activity sensor decay / spike update
    ip->scalefactor = 1.0; // Default scaling factor for this cell's AMPA synapses
    ip->goal_activity = -1; // Cell's target activity (MHz i.e. cycles per ms)
    ip->activity_integral_err = 0.0; // Integral of cell's activity divergence from target activity
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ip->gid = nextGID; nextGID += 1.0;// global identifier
    process=(int)getpid();
    CNAME[SU]="SU"; CNAME[DP]="DP"; CNAME[IN]="IN";
    if (installed==2.0 && ip->pg) { // jitcondiv was previously run
      sz=ivoc_list_count(ip->pg->ce);
      if(verbose) printf("\t**** WARNING new INTF6 created: may want to rerun jitcondiv ****\n");
    } else installed=1.0; // set or reset it
    cbsv=0x0;
  }
  ENDVERBATIM
}

PROCEDURE resetscaling () {
  VERBATIM
  ip = IDP;
  //     THE PARAMETERS IN THIS 'BLOCK' ARE ASSOCIATED WITH HOMEOSTATIC SYNAPTIC SCAING
  ip->activity = 0; // Sensor for this cell's recent activity (default 0MHz i.e. cycles per ms)
  ip->max_err = 0; // Max error value
  ip->max_scale = 100; // Max scaling factor
  ip->lastupdate = 0; // Time of last activity sensor decay / spike update
  ip->scalefactor = 1.0; // Default scaling factor for this cell's AMPA synapses
  ip->goal_activity = -1; // Cell's target activity (MHz i.e. cycles per ms)
  ip->activity_integral_err = 0.0; // Integral of cell's activity divergence from target activity
  ENDVERBATIM
}

DESTRUCTOR {
  VERBATIM { 
  free(IDP);
  }
  ENDVERBATIM
}

:** INITIAL
INITIAL { LOCAL id
  reset() 
  t0 = 0
  tg = 0
  twg = 0
  trrs = 0
  tmax=0
  pathend=-1
  pathlen=0
  VERBATIM
  { int i,ix;
  ip=IDP;
  _lid=(double)ip->id;
  ip->spkcnt=0;
  ip->blkcnt=0;
  ip->errflag=0;
  ip->pg->lastspk[ip->id]=-1;
  for (i=0;i<CTYN;i++){ix=cty[i]; blockcnt[ix]=spikes[ix]=AMo[ix]=NMo[ix]=GAo[ix]=AMo2[ix]=NMo2[ix]=GAo2[ix]=0;}
  if(seadsetting==3 && resetplast && ip->wgain) for(i=0;i<ip->dvt;i++) ip->wgain[i]=1.0; // reset learning
  if(seadsetting==3 && ip->pdope) for(i=0;i<ip->dvt;i++) ip->pdope[i] = -1e9; // turn off eligibility trace
  }
  ENDVERBATIM
  jrsvn=jrsvd jrtime=jrtm
  : init with vinset(0) if will turn on via a NetCon with w5=1
  if (vinflag()) { randspk() net_send(nxt,2)}
  if (recflag()) { recini() } : recini() resets for recording, cf recinit()
  if (pathbeg==id) { 
    stoprun=0 
    net_send(0,2) 
  } : send at time 0
  rebeg=0 : will reset this to restart storage for rec,wrec

  : 
  : SN - NB - SHOULD PROBABLY RESET AT LEAST SOME OF THE
  : PARAMS ASSOCIATED WITH HOMEOSTATIC SYNAPTIC SCALING HERE
  : 
  :  Store fixed value of 1/tau - users should not modify this!!
VERBATIM
  activityoneovertau = 1.0 / activitytau; //@

ENDVERBATIM
  : resetscaling()
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
  invlt = -1
  t0 = t
  tg = t
  twg = t
  trrs = t
  cbur = 0 : # bursts left to 0, just in case
  spck = 0 : spike count to 0
  refractory = 0 : 1 means cell is absolute refractory
  VTHC=VTH :set current threshold to absolute threshold value
  VTHR=VTH :set this one too to make sure it's initialized
}

VERBATIM
unsigned int GetDVIDSeedVal(unsigned int id) {
  double x[2];
  if (seadsetting==1) { 
    sead=((unsigned int)ip->id+seaddvioff)*1e6;
  } else { 
    if (seadsetting==2) printf("Warning: GetDVIDSeedVal called with wt rand turned off\n");
    x[0]=(double)id; x[1]=seaddvioff;
    sead=hashseed2(2,&x);
  }
  return sead;
}
ENDVERBATIM

: seed for divergence and delays -- not yet used
FUNCTION DVIDSeed(){
  VERBATIM
  return (double)GetDVIDSeedVal(IDP->id);
  ENDVERBATIM
}

:* NET_RECEIVE
NET_RECEIVE (wAM,wNM,wGA,wGB,wAM2,wNM2,wGA2,wflg) { LOCAL tmp,jcn,id
  INITIAL { wAM=wAM wNM=wNM wGA=wGA wGB=wGB wAM2=wAM2 wNM2=wNM2 wGA2=wGA2 wflg=0}
  : intra-burst, generate next spike as needed
VERBATIM
  id0 *ppre; int prty,poty,prin,prid,poid,ii,sy,nsyn,distal; double STDf,wgain,syw1,syw2; //@

ENDVERBATIM
  tmax=t
  VERBATIM
  if (stopoq && !qsz()) stoprun=1;
  ip=IDP; pg=ip->pg; ppre = 0x0; poid=ip->id;
  if (ip->dead) return; // this cell has died
  _ljcn=ip->jcn; _lid=ip->id;
  tpnt = _pnt; // this pnt
  if (PATHMEASURE) { // do all code for this
    if (_lflag==2 || _lflag<0) { // on the callback -- distribute to divergence list
      double idty; int i;
      if (_lflag==2) ip->flag=-1; 
      idty=(double)(FOFFSET+ip->id)+1e-2*(double)ip->type+1e-3*(double)ip->inhib+1e-4;
      for (i=0;i<ip->dvt && !stoprun;i++) if (ip->sprob[i]) {
        (*pnt_receive[ip->dvi[i]->_prop->_type])(ip->dvi[i], wts, idty); 
        _p=_pnt->_prop->param; _ppvar=_pnt->_prop->dparam; ip=IDP; // restore pointers each time
      }
      return;  // else see if destination has been reached
    } else if (_lflag!=2 && (pathtytarg==(double)ip->type || pathidtarg==(double)ip->id)) {
      if (pathend==(double)ip->id) return; // means that coming back here again
      ip->flag=(unsigned char)floor(t)+1; // type-target or id-target
      pathend=(double)ip->id; 
      pathlen=tmax+1; // tmax gives pathlength
      stoprun=1.; 
      return;
      // deadends:visited || no output  ||stopped
    } else if (ip->flag   || ip->dvt==0 || stoprun) {
      return; // inhib cell is a deadend; don't revisit anyone
    } else if (ip->inhib) {
      if (!ip->flag) ip->flag=(unsigned char)floor(t)+1;
    } else { // first callback will be from the stim
      ip->flag=(unsigned char)floor(t)+1;
   #if defined(t)
      net_send((void**)0x0, wts,tpnt,t+1.,-1.); // the callback call
  #else
      net_send((void**)0x0, wts,tpnt,1.,-1.); // the callback call
  #endif
      return;
    }
  }

  // MR: Synaptic scaling and deletion logic
  if (dynamicdel) {
    dynamicdelete(t); // Calculate probabilistically whether or not this cell should die
  }
  // SN - is the following line needed when not running synaptic scaling?
  decay_activity_sensor(t); // Allow activity sensor to decay on every update

  if (scaling) {
    if (ip->goal_activity < 0) {
      // If scaling has just been turned on, set goal activity to historical average firing rate
      // This is only meaningful if sensor has had a chance to measure correct activity over
      // a relatively long period of time, so don't call setscaling(1) until at least ~800s.
      //ip->goal_activity = get_avg_activity();
      ip->goal_activity = ip->activity; // Take current activity sensor value
      //ip->max_err = ip->goal_activity * 0.5; // Error value saturates at +- 50% of goal activity rate
    }

    if (!ip->inhib || scaleinhib) {
      // Only update if cell is not inhib OR we are scaling all I+E cells
      update_scale_factor(t); // Run synaptic scaling procedure to find scalefactor
    }  
  }
  ip->lastupdate = t; // Store time of last update
  


  if (_lflag==OK) { FLAG=OK; flag(); return; } // identify internal call with errflag
  if (_lflag<0) { callback(_lflag); return; }
  pg->eventtot+=1;

  // if(flag==0) { printf("flag==0!\n"); }
  ENDVERBATIM
VERBATIM
  if (ip->dbx>2) 
ENDVERBATIM
{ 
    pid() 
    printf("DB0: flag=%g Vm=%g",flag,VAM+VNM+VGA+RMP+AHP+VAM2+VNM2+VGA2)
    if (flag==0) { printf(" (%g %g %g %g %g %g %g)",wAM,wNM,wGA,wAM2,wNM2,wGA2,wflg) }
    printf("\n")
  }
: causes of spiking: between VTH and Vblock, random from vsp (flag 2), within burst
:** JITcon code - only meant for intra-COLUMN events
  if (flag>=FOFFSET) { : jitcon -- set up weights on the fly
    VERBATIM {
      // find type of presyn
      prid = (int)(_lflag-FOFFSET); // that correct? - if not, put prid in wts[2]
      poty=(int)ip->type;
      prty=(int)(1e2*(_lflag-floor(_lflag)));
      prin=(int)(1e3*(_lflag-floor(_lflag)-prty*1e-2)); // stuffed into this flag
      distal = ((int) (_lflag * 1e5 + 0.5)) % 2;       
      if(distal){ sy=prin?GA2:AM2; } else { sy=prin?GA:AM; }
      // if(verbose>4) printf("receive: %s->%s, prin=%d, distal=%d, sy=%d, _lflag=%.10f\n",\
      //                    CNAME[ctymap[prty]],CNAME[ctymap[poty]],prin,distal,sy,_lflag);
      STDf=_args[0]; // save value -- for short-term changes
      wgain=_args[1]; // save value -- for plasticity
      syw1=_args[2]; // save value -- for non-MATRIX weight 1 -- only used when wsetting==1
      syw2=_args[3]; // save value -- for non-MATRIX weight 2 -- only used when wsetting==1
      if(ip->dbx<-1) printf("prid%d,poid%d,wgain=%g\n",prid,poid,wgain); 
      for (ii=0;ii<=6;ii++) _args[ii]=0.; // clear _args (stores weights for later) to be safe
      if (seadsetting==3) { // plasticity mode is on
        ppre = getlp(pg->ce,prid);  // get pointer to presynaptic cell
        if(ip->dbx<-1) printf("ppre%p,pre%d->po%d,wg=%g\n",ppre,prid,ip->id,wgain);
        if(ppre->inhib) { // only care about appropriate presynaptic cells for plasticity
          if(!ISTDP && !IDOPE) ppre=0x0;
        } else {
          if(!ESTDP && !EDOPE) ppre=0x0;
        }
      }
      if(ppre) { // appropriate presynaptic cell AND plasticity mode is on
        for (ii=sy,nsyn=0;ii<sy+2;ii++) {
          if(ii==AM2 || ii==AM || ii==GA || ii==GA2) { // AMPA,GABAA plasticity factor
            if(wsetting==1.0) { // non-MATRIX weights and AMPA,GABAA plasticit
              _args[ii] = ii == sy ? syw1 * wgain : syw2 * wgain;
            } else { // MATRIX weights and AMPA/GABAA plasticity
              _args[ii]=wgain*WMAT(prty,poty,ii)*WD0(prty,poty,ii);
            }
            if(ip->dbx<-1) printf("pre%d->po%d,sy=%d,wg=%g,w=%g\n",prid,ip->id,ii,wgain,_args[ii]);
          } else { // non-AMPA/non-GABAA -->> no plasticity applied
            if(wsetting==1.0) { // non-MATRIX weights and non AMPA
              _args[ii] = ii == sy ? syw1 : syw2;
            } else { // MATRIX weights and non AMPA
              _args[ii]=WMAT(prty,poty,ii)*WD0(prty,poty,ii);
            }
          }
          nsyn+=(_args[ii]>0.);
        }
      } else { // no plasticity applied
        if(wsetting==1.0) { // non-MATRIX weights
          _args[sy+0] = syw1;
          _args[sy+1] = syw2;
          nsyn = (_args[sy+0]>0.) + (_args[sy+1]>0.);
        } else { // MATRIX weights
          for (ii=sy,nsyn=0;ii<sy+2;ii++) nsyn+=((_args[ii]=WMAT(prty,poty,ii)*WD0(prty,poty,ii))>0.);
        }
      }
      if (nsyn==0) return; //return for 0-weight events, before changing state vars or Vm

      // *** Do synaptic scaling
      if (scaling) {
        for (ii=sy,nsyn=0;ii<sy+2;ii++) {
          if (!ip->inhib) {
            // Scale E cell
            if (ii==AM2 || ii==AM) { // || ii==NM || ii == NM2) {
              // Scale AMPA receptors by scalefactor (Turrigiano, 2008)
              _args[ii] *= ip->scalefactor;
            }
            if (ii==GA || ii==GA2) {
              // Scale GABA receptors by 1/scalefactor to model BDNF (Chandler and Grossberg, 2012)
              _args[ii] *= 1 / ip->scalefactor;
            }
          } else {
            // Scale I cell
            // Scaling has opposite effects on I cells (if scaling is enabled for I cells)
            if (ii==AM2 || ii==AM) { // || ii==NM || ii == NM2) {
              // Scale I-cell AMPA receptors by 1/scalefactor
              _args[ii] *= 1 / ip->scalefactor;
            }
            if (ii==GA || ii==GA2) {
              // Scale I-cell GABA receptors by scalefactor
              _args[ii] *= ip->scalefactor;
            }

          }
        }
      }
      // *** Done synaptic scaling

      if (seadsetting==3) { // empty 'if' to skip next clause
      } else if (seadsetting!=2) { // not fixed weights
        if (seadsetting==1) {
          sead=(unsigned int)(floor(_lflag)*ip->id*seedstep); // all integers
        } else { // hash on presynaptic id+FOFFSET,poid,seedstep
          hsh[0]=floor(_lflag); hsh[1]=(double)ip->id; hsh[2]=seedstep;
          sead=hashseed2(3,&hsh); // hsh[] is just scratch pad
        }
        mcell_ran4(&sead, &_args[sy], 2, 1.);
        for (ii=sy;ii<sy+2;ii++) { // scale appropriately; 
          _args[ii]=2*WVAR*(_args[ii]+0.5/WVAR-0.5)*WMAT(prty,poty,ii)*WD0(prty,poty,ii);
        }
      }
    }
    ENDVERBATIM
VERBATIM
    if (ip->dbx>2) 
ENDVERBATIM
{ 
      pid() 
      printf("DF: flag=%g Vm=%g",flag,VAM+VNM+VGA+RMP+AHP+VAM2+VNM2+VGA2)
      printf(" (%g %g %g %g %g %g %g)",wAM,wNM,wGA,wAM2,wNM2,wGA2,wflg)
      printf("\n")
    }
:** mid-burst
  } else if (flag==4) { 
    cbur=cbur-1  : count down the spikes
    if (cbur>0) { 
      net_send(tbur,4) 
    } else { : end of burst
      refractory = 1      : signal that this cell is in refractory period
      net_send(refrac-AHP*AHP2REF, 3) : send event for end of refractory
    }
    tmp=t
VERBATIM
    if (ip->jttr) 
ENDVERBATIM
{ tmp= t+jttr()/10 } 
    if (jcn) { 
      jitcon(tmp)
VERBATIM
      if(ip->out) 
ENDVERBATIM
{ net_event(tmp) } 
    } else { net_event(tmp) }
VERBATIM
    spikes[ip->type]++; //@

ENDVERBATIM
    spck=spck+1
VERBATIM
    if (ip->dbx>0) 
ENDVERBATIM
{ pid() printf("DBA: mid-burst event at %g, %g\n",tmp,cbur) } 
VERBATIM
    if (ip->record) 
ENDVERBATIM
{ recspk(tmp) } 
VERBATIM
    if (ip->wrec) 
ENDVERBATIM
{ wrecord(t) } 
VERBATIM
    return; //@ done

ENDVERBATIM
    : start reading random spike times (or burst times) from vsp vector pointer
    : this is signaled externally from a netstim with wflg=1, will turn off on next stim 
    : (NB wflg used in completely different context for GABAB) ?? is this still true ??
    : this is bad -- should use a special netcon that just handles signals
  } else if (flag==0 && wflg==1) {
VERBATIM
    ip->input=1; //@

ENDVERBATIM
    wflg=2 : set flag to turn off next time an external event comes from here
    randspk() 
    net_send(nxt,2)
VERBATIM
    return; //@ done

ENDVERBATIM
  } else if (flag==0 && wflg==2) { : flag to stop random spikes
VERBATIM
    ip->input=0; //@ inputs that are read from a vector of times -- see randspk()

ENDVERBATIM
    wflg=1  : flag to turn on next time
VERBATIM
    return; //@ done

ENDVERBATIM
  }
  : update state variables
VERBATIM
  if (ip->record) 
ENDVERBATIM
{ record() } 
VERBATIM
  if (ip->wrec) 
ENDVERBATIM
{ wrecord(1e9) } 
:** update state variables: VAM, VNM, VGA
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
  if (flag==0 || flag>=FOFFSET) { 

    : AMPA Erev=0 (0-RMP==65 mV above rest)
    if (wAM>0) {
      if (STDAM==0) { VAM = VAM + wAM*(1-Vm/EAM)
      } else        { VAM = VAM + (1-STDAM*STDf)*wAM*(1-Vm/EAM) }
      if (VAM>EAM) { 
VERBATIM
        AMo[ip->type]++; //@

ENDVERBATIM
      } else if (VAM<0) { VAM=0 }
    }
    if (wAM2>0) { : AMPA from distal dends
      if (STDAM==0) { VAM2 = VAM2 + wAM2*(1-Vm/EAM)
      } else        { VAM2 = VAM2 + (1-STDAM*STDf)*wAM2*(1-Vm/EAM) }
      if (VAM2>EAM) { 
VERBATIM
        AMo2[ip->type]++; //@

ENDVERBATIM
      } else if (VAM2<0) { VAM2=0 }
    }
    : NMDA; Mg effect based on total activation in rates()
    if (wNM>0 && VNM<ENM) { 
      if (STDNM==0) { VNM = VNM + wNM*rates(RMP+Vm)*(1-Vm/ENM) 
      } else        { VNM = VNM + (1-STDNM*STDf)*wNM*rates(RMP+Vm)*(1-Vm/ENM) }
      if (VNM>ENM) { 
VERBATIM
        NMo[ip->type]++; //@

ENDVERBATIM
      } else if (VNM<0) { VNM=0 }
    }
    if (wNM2>0 && VNM2<ENM) { : NMDA from distal dends
      if (STDNM==0) { VNM2 = VNM2 + wNM2*rates(RMP+Vm)*(1-Vm/ENM)
      } else        { VNM2 = VNM2 + (1-STDNM*STDf)*wNM2*rates(RMP+Vm)*(1-Vm/ENM) }
      if (VNM2>ENM) { 
VERBATIM
        NMo2[ip->type]++; //@

ENDVERBATIM
      } else if (VNM2<0) { VNM2=0 }
    }
    : GABAA , GABAA2 : note that all wts are positive
    if (wGA>0 && VGA>EGA) { : the neg here gives the inhibition
      if (STDGA==0) {  VGA = VGA - wGA*(1-Vm/EGA) 
      } else {         VGA = VGA - (1-STDGA*STDf)*wGA*(1-Vm/EGA) }
      if (VGA<EGA) { 
VERBATIM
        GAo[ip->type]++; //@

ENDVERBATIM
VERBATIM
        if (ip->dbx>2) 
ENDVERBATIM
{ 
          pid() printf("DB0A: flag=%g Vm=%g",flag,VAM+VNM+VGA+RMP+AHP+VAM2+VNM2+VGA2)
          if (flag==0) { printf(" (%g %g %g %g %g %g)",wGA,EGA,VGA,Vm,AHP,STDf) }  
VERBATIM
          printf("\nAA:%d:%d\n\n",GAo[ip->type],ip->type); //@ 

ENDVERBATIM
        }
      } else if (VGA>0) { VGA=0 } : if want reversal of VGA need to also edit above
    }
    if (wGA2>0 && VGA2>EGA) { : the neg here gives the inhibition, GABAA2, inputs further from soma
      if (STDGA==0) {  VGA2 = VGA2 - wGA2*(1-Vm/EGA)
      } else {         VGA2 = VGA2 - (1-STDGA*STDf)*wGA2*(1-Vm/EGA) }
      if (VGA2<EGA) { 
VERBATIM
        GAo2[ip->type]++; //@

ENDVERBATIM
VERBATIM
        if (ip->dbx>2) 
ENDVERBATIM
{ 
          pid() printf("DB0A: flag=%g Vm=%g",flag,VAM+VNM+VGA+RMP+AHP+VAM2+VNM2+VGA2)
          if (flag==0) { printf(" (%g %g %g %g %g %g)",wGA2,EGA,VGA2,Vm,AHP,STDf) }  
VERBATIM
          printf("\nAA:%d:%d\n\n",GAo2[ip->type],ip->type); //@ 

ENDVERBATIM
        }
      } else if (VGA2>0) { VGA2=0 } : if want reversal of VGA2 need to also edit above
    }
:*** modulated interval firing; cf invlfire.mod
VERBATIM
    if (ip->invl0) 
ENDVERBATIM
{ 
      Vm = RMP+VAM+VNM+VGA+AHP+VAM2+VNM2+VGA2
      if (Vm>0)   {Vm= 0 }
      if (Vm<-90) {Vm=-90}
      if (invlt==-1) { : activate for first time
        if (Vm>RMP) {
          oinvl=invl
          invlt=t
          net_send(invl,1) 
        }
      } else {
        tmp=shift(Vm)
        if (tmp!=0)  {
          net_move(tmp) 
          if (id()<prnum) {
            pid() printf("**** MOVE t=%g to %g Vm=%g %g,%g\n",t,tmp,Vm,invlt,oinvl) }
        }
      }      
    }
  } else if (flag==1) { : modulated interval firing; cf invlfire.mod
    : Vm=RMP+VAM+VNM+VGA+AHP+VAM2+VNM2+VGA2
    if (WINV<0) { 
      if (jcn) { 
        jitcon(t)
VERBATIM
        if(ip->out) 
ENDVERBATIM
{ net_event(t) } 
      } else { net_event(t) } : bypass activation calculation
VERBATIM
      spikes[ip->type]++; //@

ENDVERBATIM
      spck=spck+1
VERBATIM
      if (ip->dbx>0) 
ENDVERBATIM
{pid() printf("DBC: interval event\n")}  
VERBATIM
      if (ip->record) 
ENDVERBATIM
{ recspk(t) } 
VERBATIM
      if (ip->wrec) 
ENDVERBATIM
{ wrecord(t) } 
    } else {
      tmp = WINV*(1-Vm/EAM)
      VAM = VAM + tmp :: activate interval depolarization
    }
    oinvl=invl
    invlt=t
    net_send(invl,1) 
  } else if (flag==2) { :** flag==2 -- read off external vec (vsp) for next random spike time or single from shock()
VERBATIM
    if (ip->dbx>1) 
ENDVERBATIM
{pid() printf("DBBa: randspk called: %g,%g\n",WEX,nxt)} 
    if (WEX>1e8) { : super-threshold event
      if (jcn) { 
        jitcon(t)
VERBATIM
        if(ip->out) 
ENDVERBATIM
{ net_event(t) } 
      } else { net_event(t) } : bypass activation calculation
VERBATIM
      spikes[ip->type]++; //@

ENDVERBATIM
      spck=spck+1
VERBATIM
      if (ip->dbx>0) 
ENDVERBATIM
{pid() printf("DBB: randspk event @ t=%g\n",t)} 
VERBATIM
      if (ip->record) 
ENDVERBATIM
{ recspk(t) } 
VERBATIM
      if (ip->wrec) 
ENDVERBATIM
{ wrecord(t) } 
    } else if (WEX>0) { : excitatory input
      if(EXSY==AM) {
        tmp = WEX*(1-Vm/EAM)
        VAM = VAM + tmp
      } else if(EXSY==AM2) {
        tmp = WEX*(1-Vm/EAM)
        VAM2 = VAM2 + tmp
      } else if(EXSY==NM) {
        tmp = rates(RMP+Vm)*WEX*(1-Vm/ENM)
        VNM = VNM + tmp
      } else if(EXSY==NM2) {
        tmp = rates(RMP+Vm)*WEX*(1-Vm/ENM)
        VNM2 = VNM2 + tmp
      }
    } else if (WEX<0 && WEX!=-1e9) { : inhibitory input
      if(EXSY==GA) {
        tmp = WEX*(1-Vm/EGA)
        VGA = VGA + tmp
      } else { :GA2
        tmp = WEX*(1-Vm/EGA)
        VGA2 = VGA2 + tmp
      }
    }
    if (WEX!=-1e9) { : code for single shock
      randspk() : will set WEX for next time
VERBATIM
      if (ip->input) 
ENDVERBATIM
{ net_send(nxt,2) } 
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
    if (!ip->vbr && Vm>Vblock) {//@ do nothing

ENDVERBATIM
VERBATIM
      ip->blkcnt++; blockcnt[ip->type]++; return; }//@

ENDVERBATIM
    AHP = AHP - ahpwt
    tmp=t
    : note that jtt indicates jitter while jit indicates 'just-in-time'
VERBATIM
    if (ip->jttr) 
ENDVERBATIM
{ tmp= t+jttr() }  
VERBATIM
    //printf("spk t = %g\n",_ltmp); //@

ENDVERBATIM
VERBATIM
    //printf("a ip->pg->lastspk[%d]=%g\n",ip->id,ip->pg->lastspk[ip->id]); //@

ENDVERBATIM
VERBATIM
    raise_activity_sensor(t); //@ Update activity sensor

ENDVERBATIM
VERBATIM
    ip->pg->lastspk[ip->id]=_ltmp; //@

ENDVERBATIM
VERBATIM
    //printf("b ip->pg->lastspk[%d]=%g\n",ip->id,ip->pg->lastspk[ip->id]); //@

ENDVERBATIM
    if (jcn) { 
      jitcon(tmp)
VERBATIM
      if(ip->out) 
ENDVERBATIM
{ net_event(tmp) } 
    } else { net_event(tmp) } 
VERBATIM
    spikes[ip->type]++; //@

ENDVERBATIM
    spck=spck+1            
VERBATIM
    if (ip->dbx>0) 
ENDVERBATIM
{pid() printf("DBD: %g>VTH(%g) event at %g (STDf=%g)\n",Vm,VTHC,tmp,STDf)} 
VERBATIM
    if (ip->record) 
ENDVERBATIM
{ recspk(tmp) } 
VERBATIM
    if (ip->wrec) 
ENDVERBATIM
{ wrecord(tmp) } 
    if(incRR) { : additive
      VTHC=VTHC+RRWght*(Vblock-VTH):increase threshold for relative refrac. period. NB: RRWght can be < 0
      if(VTHC > Vblock) {VTHC=Vblock} else if(VTHC < RMP) {VTHC=RMP}
    } else { : non-additive
      VTHC=VTH+RRWght*(Vblock-VTH):increase threshold for relative refrac. period. NB: RRWght can be < 0
    }
    VTHR=VTHC :starting thresh value for relative refrac period, keep track of it
    refractory = 1 : abs. refrac on = don't allow any more spikes/bursts to begin (even for IB cells)

    if(seadsetting==3) { : apply learning rule
      if(plaststartT<0 || plastendT<0 || (t>=plaststartT && t<=plastendT)) { : make sure plasticity on now
VERBATIM
        if(ip->dbx<-1) printf("%d@%g applying plasticity\n",ip->id,ip->pg->lastspk[ip->id]); //@

ENDVERBATIM
VERBATIM
        if(ESTDP) applyEXSTDP(ip,ip->pg->lastspk[ip->id]); //@

ENDVERBATIM
VERBATIM
        if(ISTDP) applyIXSTDP(ip,ip->pg->lastspk[ip->id]); //@

ENDVERBATIM
VERBATIM
        if(EDOPE) applyEDOPE(ip,ip->pg->lastspk[ip->id]); //@

ENDVERBATIM
VERBATIM
        if(IDOPE) applyIDOPE(ip,ip->pg->lastspk[ip->id]); //@

ENDVERBATIM
      }
    }

    if (nbur>1) { 
      cbur=nbur-1 net_send(tbur,4) : this is main source of burst events - A.P. firing with bursting
VERBATIM
      return; //@ done

ENDVERBATIM
    } 
VERBATIM
  if (ip->vbr && Vm>Vblock) 
ENDVERBATIM
{ 
      net_send(Vbrefrac,3) 
VERBATIM
     if (ip->dbx>0) 
ENDVERBATIM
{pid() printf("DBE: %g %g\n",Vbrefrac,Vm)} 
VERBATIM
      return; //@ done

ENDVERBATIM
    }
    net_send(refrac-AHP*AHP2REF, 3) :event for end of abs. refrac., sent separately for IB cells @ end of burst
  }
}

:* ancillary functions
:** jitcon() creates divergence and delays from rand seed
: jcn flags:
: 0 NetCons                            jcn==0
: 3 Jitcon without jitevent            jcn==3 -- eliminated after v669
: 2 Jitcon with callback               jcn==2 -- NOT DEBUGGED
: 1 Jitcon with callback with pointers jcn==1
PROCEDURE jitcon (tm) {
  VERBATIM {
  double mindel, randel, idty, *x; int prty, poty, i, j, k, dv; 
  Point_process *pnt; void* voi;
  // qsz = nrn_event_queue_stats(stt);
  // if (qsz>=qlimit) { printf("qlimit %g exceeded at t=%g\n",qlimit,t); qlimit*=2; }
  ip=IDP; pg=ip->pg;
  if(verbose>1) printf("col %d , ip %p, pg %p\n",ip->col,ip,pg);
  if (!pg) {printf("No network defined -- must run jitcondiv()\n"); hxe();}
  ip->spkcnt++; // jitcon() called from NET_RECEIVE which sets ip
  if (pg->jrj<pg->jrmax) {  // record spike time and cell ID
    pg->jrid[pg->jrj]=(double)ip->id; pg->jrtv[pg->jrj]=_ltm;
    pg->jrj++;
  } else if (wf2 && pg->jrmax) spkoutf2(); // saving spike times
  pg->jri++;  // keep track of number of spikes
  if (jrtm>0) {
    if (t>jrtime) {
      jrtime+=jrtm;
      spkstats2(1.);
    }
  } else if (jrsvd>0 && pg->jri>jrsvn) { 
    jrsvn+=jrsvd; printf("t=%.02f %ld ",t,ip->pg->jri);
    spkstats2(1.);
  }
  prty=(int)ip->type;
  if (ip->jcn==1) if (ip->dvt>0) {  // first callback
      #if defined(t)
    if (ip->jcn==1) if (ip->dvt>0) net_send((void**)0x0, wts,tpnt,t+ip->del[0],-1.);
      #else
    if (ip->jcn==1) if (ip->dvt>0) net_send((void**)0x0, wts,tpnt,ip->del[0],-1.);
      #endif
  }
  }   
  ENDVERBATIM  
}

: call spkstat from hoc to set global tf if desired for spkstats to file
PROCEDURE spkstats () {
VERBATIM {
  if (ifarg(1)) tf=hoc_obj_file_arg(1); else tf=0x0;
}
ENDVERBATIM
}

: spkoutf() use wf2 for output of indices and times
PROCEDURE spkoutf () {
VERBATIM {
  if (ifarg(2)) {
    wf1=hoc_obj_file_arg(1); // index file
    wf2=hoc_obj_file_arg(2);
  } else if (wf1 != 0x0) {
    spkoutf2();
    wf1=(FILE*)0x0; wf2=(FILE*)0x0;
  }
}
ENDVERBATIM
}

VERBATIM
static int spkoutf2 () {
    fprintf(wf1,"//b9 -2 t%0.2f %ld %ld\n",t/1e3,pg->jrj,ftell(wf2));
    fwrite(pg->jrtv,sizeof(double),pg->jrj,wf2); // write times
    fwrite(pg->jrid,sizeof(double),pg->jrj,wf2); // write id
    fflush(wf1); fflush(wf2);
    pg->jrj=0;
}
ENDVERBATIM

PROCEDURE callhoc () {
  VERBATIM
  if (ifarg(1)) {
    cbsv=hoc_lookup(gargstr(1));
  } else {
    cbsv=0x0;
  }
  ENDVERBATIM
}

: flag 1 means print it to a file, 2 means to both places
PROCEDURE spkstats2 (flag) {
VERBATIM {
  int i, ix, flag; double clk;
  ip=IDP; pg=ip->pg;
  flag=(int)(_lflag+1e-6);
  clk=clock()-savclock; savclock=clock();
  if (cbsv) hoc_call_func(cbsv,0);
  if (tf) fprintf(tf,"t=%.02f;%ld(%g) ",t,pg->jri,clk/1e6); else {
    printf("t=%.02f;%ld(%g) ",t,pg->jri,clk/1e6); }
  for (i=0;i<CTYN;i++) {
    ix=cty[i];
    pg->spktot+=spikes[ix];
    if (tf) {
      fprintf(tf,"%s:%d/%d:%d;%d;%d;%d;%d;%d ",CNAME[i],spikes[ix],\
              blockcnt[ix],AMo[ix],NMo[ix],GAo[ix],AMo2[ix],NMo2[ix],GAo2[ix]);
    } else {
      printf("%s:%d/%d:%d;%d;%d;%d;%d;%d ",CNAME[i],spikes[ix],blockcnt[ix],\
             AMo[ix],NMo[ix],GAo[ix],AMo2[ix],NMo2[ix],GAo2[ix]);
    }
    spck=0;
    blockcnt[ix]=spikes[ix]=0;
    AMo[ix]=NMo[ix]=GAo[ix]=AMo2[ix]=NMo2[ix]=GAo2[ix]=0;
  }
  if (tf && flag==2) {  fprintf(tf,"\nt=%g tot_spks: %ld; tot_events: %ld\n",t,pg->spktot,pg->eventtot); 
  } else if (flag==2) {  printf("\ntotal spikes: %ld; total events: %ld\n",pg->spktot,pg->eventtot); 
  } else if (tf) fprintf(tf,"\n"); else printf("\n");
}
ENDVERBATIM
}

PROCEDURE oobpr () {
VERBATIM {
  int i,ix;
  for (i=0;i<CTYN;i++){ 
    ix=cty[i];
    printf("%d:%d/%d:%d;%d;%d;%d;%d;%d ",ix,spikes[ix],blockcnt[ix],\
           AMo[ix],NMo[ix],GAo[ix],AMo2[ix],NMo2[ix],GAo2[ix]);
  }
  printf("\n");
}
ENDVERBATIM
}

PROCEDURE callback (fl) {
  VERBATIM {
  int i; double idty, idtflg, del0, ddel; id0 *jp; Point_process *upnt; // these must be local
  i=(unsigned int)((-_lfl)-1); // -1,-2,-3 -> 0,1,2
  jp=IDP; upnt=tpnt; del0=jp->del[i]; ddel=0.;
  idty=(double)(FOFFSET+jp->id)+1e-2*(double)jp->type+1e-3*(double)jp->inhib+1e-4;
  while (ddel<=DELMIN) { // check if this del is worth waiting, else just send now
    if (Vblock<VTHC) { 
      wts[0]=0; // send [0,1] for STD
    } else if(STDAM || STDNM || STDGA) { // STDf=(1-STD) , ONLY SET wts[0] WHEN SHORT-TERM FACIL. ON
                                                         //NB: WTS IS TOO OVERUSED, CONFUSING!!!!!!!!!!!
                                                         //if anyone uses STD they should make sure doesn't
                                                         //cause problems in wts, _args !!!
      wts[0]=(VTHC-VTH)/(Vblock-VTH); // just send [0,1] for STD
    }
    wts[1]=0.0; // default is no plasticity gain
    if(seadsetting==3) { // check if should send plasticity gain
      if(jp->inhib) {
        if(ISTDP || IDOPE) wts[1]=jp->wgain[i];
      } else {
        if(ESTDP || EDOPE) wts[1]=jp->wgain[i];
      }
    }
    if(wsetting==1.0 && jp->syw1 && jp->syw2) {wts[2]=jp->syw1[i]; wts[3]=jp->syw2[i]; } // non-MATRIX weights?
    idtflg = idty + (1e-5 * jp->syns[i]);
    // if(1) printf("s = %g : flg = %.10f\n",(1e-5*jp->syns[i]),idtflg);
    if (jp->sprob[i]) (*pnt_receive[jp->dvi[i]->_prop->_type])(jp->dvi[i], wts, idtflg); 
    _p=upnt->_prop->param; _ppvar=upnt->_prop->dparam; // restore pointers
    i++;
    if (i>=jp->dvt) return 0; // ran out
    ddel=jp->del[i]-del0;   // delays are relative to event; use difference in delays
  }
  // skip over pruned outputs and dead cells:
  while (i<jp->dvt && (!jp->sprob[i] || (*(id0**)&(jp->dvi[i]->_prop->dparam[2]))->dead)) i++;
  if (i<jp->dvt) {
    ddel= jp->del[i] - del0;;
  #if defined(t)
    net_send((void**)0x0, wts,upnt,t+ddel,(double) -(i+1)); // next callback
  #else
    net_send((void**)0x0, wts,upnt,ddel,(double) -(i+1)); // next callback
  #endif
  }
  } 
  ENDVERBATIM
}

: DEAD_DIV not checked in mkdvi()
: mkdvi() create the connectivity vectors for a random network
PROCEDURE mkdvi () {
VERBATIM {
  int i,j,k,prty,poty,dv,dvt,dvii; double *x, *db, *dbs; 
  Object *lb;  Point_process *pnnt, **da, **das;
  ip=IDP; pg=ip->pg;//this should only be called after jitcondiv()
  if (ip->dead) return 0;
  prty=ip->type;
  sead=GetDVIDSeedVal(ip->id);//seed for divergence and delays
  for (i=0,k=0,dvt=0;i<CTYN;i++) { // dvt gives total divergence
    poty=cty[i];
    dvt+=DVG(prty,poty);
  }
  da =(Point_process **)malloc(dvt*sizeof(Point_process *));
  das=(Point_process **)malloc(dvt*sizeof(Point_process *)); // das,dbs for after sort
  db =(double *)malloc(dvt*sizeof(double)); // delays
  dbs=(double *)malloc(dvt*sizeof(double)); // delays
  for (i=0,k=0,dvii=0;i<CTYN;i++) { // cell types in cty[]
    poty=cty[i];
    dv=DVG(prty,poty);
    if (dv>0) {
      sead+=dv;
      if (dv>dscrsz) {
        printf("B:Divergence exceeds dscrsz: %d>%d for %d->%d\n",dv,dscrsz,prty,poty); hxe(); }
      mcell_ran4(&sead, dscr ,  dv, pg->ixe[poty]-pg->ix[poty]+1);
      for (j=0;j<dv;j++) {
        if (!(lb=ivoc_list_item(pg->ce,(unsigned int)floor(dscr[j]+pg->ix[poty])))) {
          printf("INTF6:callback %g exceeds %d for list ce\n",floor(dscr[j]+pg->ix[poty]),pg->cesz); 
          hxe(); }
        pnnt=(Point_process *)lb->u.this_pointer;
        da[j+dvii]=pnnt;
      }
      mcell_ran4(&sead, dscr , dv, 2*DELD(prty,poty));
      for (j=0;j<dv;j++) {
        db[j+dvii]=dscr[j]+DELM(prty,poty)-DELD(prty,poty); // +/- DELD
        if (db[j+dvii]<0) db[j+dvii]=-db[j+dvii];
      }
      dvii+=dv;
    }
  }
  gsort2(db,da,dvt,dbs,das);
  ip->del=dbs;   ip->dvi=das;   ip->dvt=dvt; ip->syns=(char*)calloc(dvt,sizeof(char));
  ip->sprob=(unsigned char *)malloc(dvt*sizeof(char *)); // release probability
  for (i=0;i<dvt;i++) ip->sprob[i]=1; // start out with all firing
  free(da); free(db); // keep das,dbs which are assigned to ip->dvi bzw ip->del
  }
ENDVERBATIM
}

:* paths
PROCEDURE patha2b () {
  VERBATIM
  int i; double idty, *x; static Point_process *_pnt; static id0 *ip0;
  ip=IDP; pg=ip->pg;
  pathbeg=*getarg(1); pathidtarg=*getarg(2);
  pathtytarg=-1;  PATHMEASURE=1; pathlen=stopoq=0;
  for (i=0;i<pg->cesz;i++) { lop(pg->ce,i); 
    if ((i==pathbeg || i==pathidtarg) && qp->inhib) {
      pid(); printf("Checking to or from inhib cell\n" ); hxe(); }
    qp->flag=qp->vinflg=0; 
  }
  hoc_call_func(hoc_lookup("finitialize"), 0);
  cvode_fadvance(1000.0); // this call will not return
  ENDVERBATIM
}

:* paths
: pathgrps(vpre,vpos,vout) finds path lengths from pres to posts
FUNCTION pathgrps () {
  VERBATIM
  int i,j,k,na,nb,flag; double idty,*a,*b,*x,sum; static Point_process *_pnt; static id0 *ip0;
  Symbol* s; char **pfl;
  ip=IDP; pg=ip->pg;
  x=0x0;
  s=hoc_lookup("finitialize");
  if (ifarg(2)) {
    na=vector_arg_px(1,&a);
    nb=vector_arg_px(2,&b);
    if (ifarg(3)) x=vector_newsize(vector_arg(3),na*nb);
  } else {
    na=nb=pg->cesz;  // may want to put output into an unsigned char eventually
    if (ifarg(1)) x=vector_newsize(vector_arg(1),na*nb);
  }
  // if (scrsz<cesz) scrset(cesz); 
  pfl = (char **)malloc(pg->cesz * (unsigned)sizeof(char *));
  for (i=0;i<pg->cesz;i++) { lop(pg->ce,i); scr[i]=qp->inhib; pfl[i]=&qp->flag; }
  pathtytarg=-1;  PATHMEASURE=1; pathlen=stopoq=0;
  for (k=0,sum=0;k<na;k++) {
    pathbeg=a[k]; 
    if (scr[(int)pathbeg]) { 
      if (x) for (j=0;j<nb;j++) x[k*nb+j]=0.;
      continue;
    }
    for (j=0;j<nb;j++) { 
      pathidtarg=b[j]; 
      if (scr[(int)pathidtarg]) { if (x) x[k*nb+j]=0.; 
        continue;
      }
      // for (i=0;i<cesz;i++) {lop(ce,i); qp->flag=0;}
      for (i=0;i<pg->cesz;i++) *pfl[i]=0;
      hoc_call_func(s, 0);
      cvode_fadvance(1000.0); // this call will not return
      sum+=pathlen;
      if (x) x[k*nb+j]=pathlen;
    }
  }
  PATHMEASURE=0;
  free(pfl);
  _lpathgrps=sum/na/nb;
  ENDVERBATIM
}

:* intf.getdvi() get divergence (& optionally associated vectors)
: intf.getdvi(index_vec,delay_vec[,prob_vec,wt1vec,wt2vec,distalsyns,wgain]) -- need both wt1vec and wt2vec
: index = postsynaptic IDs, delay = delay, prob = probability of firing, wt1/wt2 are base weights,
: distalsyns=distal/prox synapse,wgain is multiplier from plasticity/learning
: other forms of this function call:
:  intf.getdvi(getactive.flag,vecs) with flag==1 return types  instead of ids
:  intf.getdvi(getactive.flag,vecs) with flag==2 then sum up number of each type
:  intf.getdvi(getactive.flag,vecs) with flag==3 return column instead of ids
:  with getactive flag ignores pruned connections ie 1.2 is getactive==1 and flag==2
FUNCTION getdvi () {
  VERBATIM 
  {
    int i,j,k,iarg,av1,a2,a3,a4,a6,a7,dvt,getactive=0,idx=0,*pact,prty,poty,sy,ii; 
    double *dbs, *x,*x1,*x2,*x3,*x4,*x5,*x6,*x7,idty,y[2],flag;
    void* voi, *voi2,*voi3; Point_process **das;
    ip=IDP; pg=ip->pg;
    getactive=a2=a3=a4=0;
    if (ip->dead) return 0.0;
    dvt=ip->dvt; 
    dbs=ip->del;   das=ip->dvi;
    _lgetdvi=(double)dvt; 
    if (!ifarg(1)) return _lgetdvi; // just return the divergence value
    iarg=1;
    if (hoc_is_double_arg(iarg)) {
      av1=2;
      flag=*getarg(iarg++);
      getactive=(int)flag;
      flag-=(double)getactive; // flag is in the decimal place 1.2 has flag of 2
      if (flag!=0) flag=floor(flag*10+hoc_epsilon); // avoid roundoff error
    } else av1=1; // 1st vector arg
    //just get active postsynapses (not dead and non pruned)
    voi=vector_arg(iarg++); 
    if (flag==2) { x1=vector_newsize(voi,CTYPi); for (i=0;i<CTYPi;i++) x1[i]=0;
    } else x1=vector_newsize(voi,dvt);
    if (ifarg(iarg)) { voi=vector_arg(iarg++); x2=vector_newsize(voi,dvt);  a2=1; }
    if (ifarg(iarg)) { voi=vector_arg(iarg++); x3=vector_newsize(voi,dvt); a3=1;}
    if (ifarg(iarg)) { // need 2 weight vecs for AM/NM or GA/GB
      voi=vector_arg(iarg++); x4=vector_newsize(voi,dvt); a4=1;
      voi=vector_arg(iarg++); x5=vector_newsize(voi,dvt);
    }//for prox vs dist syn vec
    if (ifarg(iarg)) { voi=vector_arg(iarg++); x6=vector_newsize(voi,dvt); a6=1;} else a6=0;
    if (ifarg(iarg)) { voi=vector_arg(iarg++); x7=vector_newsize(voi,dvt); a7=1;} else a7=0;//plasticity wgain
    idty=(double)(FOFFSET+ip->id)+1e-2*(double)ip->type+1e-3*(double)ip->inhib+1e-4;
    prty=ip->type; sy=ip->inhib?GA:AM;
    for (i=0,j=0;i<dvt;i++) {
      qp=*((id0**) &((das[i]->_prop->dparam)[2])); // #define sop	*_ppvar[2].pval
      if (getactive && (qp->dead || ip->sprob[i]==0)) continue;
      if (flag==1) { x1[j]=(double)qp->type; 
      } else if (flag==2) { x1[qp->type]++; 
      } else if (flag==3) { x1[j]=(double)qp->col; 
      } else x1[j]=(double)qp->id;
      if (a2) x2[j]=dbs[i];
      if (a3) x3[j]=(double)ip->sprob[i];
      if (a4) {
        if(ip->inhib){sy=ip->syns[i]?GA2:GA;} else {sy=ip->syns[i]?AM2:AM;} 
        poty = qp->type;
        if(wsetting==1) { // non-wmat weights
          y[0]=ip->syw1[i]; y[1]=ip->syw2[i];
        } else {
          if (seadsetting==2 || seadsetting==3) { // no randomization [or plasticity (also no randomization)]
            for(ii=0;ii<2;ii++) y[ii]=WMAT(prty,poty,sy+ii)*WD0(prty,poty,sy+ii);
          } else {
            if (seadsetting==1) { // old sead setting
              sead=(unsigned int)(FOFFSET+ip->id)*qp->id*seedstep; 
            } else { // hashed sead setting
              hsh[0]=(double)(FOFFSET+ip->id); hsh[1]=(double)(qp->id); hsh[2]=seedstep;
              sead=hashseed2(3,&hsh); 
            }
            mcell_ran4(&sead, y, 2, 1.);
            for(ii=0;ii<2;ii++) {
              y[ii]=2*WVAR*(y[ii]+0.5/WVAR-0.5)*WMAT(prty,poty,sy+ii)*WD0(prty,poty,sy+ii); }
          }
        }
        x4[j]=y[0]; x5[j]=y[1];
      }
      if (a6) x6[j] = ip->syns[i];  // distal / prox syns
      if (a7 && ip->wgain)x7[j]=ip->wgain[i];//weight gain from plasticity (stored separately from starting weight)
      j++;
    }
    if (flag!=2 && j!=dvt) for (i=av1;i<iarg;i++) vector_resize(vector_arg(i),j);
    _lgetdvi=(double)j; 
  }
  ENDVERBATIM
}

: intf.getconv(getactive.flag,vecs) with flag==1 return types instead of ids
: flags getactive.flag flag==2 then sum up number of each type
FUNCTION getconv () {
VERBATIM 
{
  int iarg,i,j,k,dvt,sz,prfl,getactive; double *x,flag;
  void* voi; Point_process **das; id0 *pp;
  ip=IDP; pg=ip->pg; // this should only be called after jitcondiv()
  sz=ip->dvt; //  // assume conv similar to div
  getactive=0;
  if (ifarg(iarg=1) && hoc_is_double_arg(iarg)) {
    flag=*getarg(iarg++);
    getactive=(int)flag;
    flag-=(double)getactive; // flag is in the decimal place 1.2 has flag of 2
    if (flag!=0) flag=floor(flag*10+hoc_epsilon);
  }
  if (!ifarg(iarg)) prfl=0; else { prfl=1;
    voi=vector_arg(iarg); 
    if (flag==2.) { x=vector_newsize(voi,CTYPi); for (i=0;i<CTYPi;i++) x[i]=0;
    } else x=vector_newsize(voi,sz); 
  } 
  for (i=0,k=0; i<pg->cesz; i++) {
    lop(pg->ce,i);
    if (getactive && qp->dead) continue;
    dvt=qp->dvt; das=qp->dvi;
    for (j=0;j<dvt;j++) {
      if (getactive && qp->sprob[j]==0) continue;
      if (ip==*((id0**) &((das[j]->_prop->dparam)[2]))) {
        if (prfl) {
          if (flag!=2.0 && k>=sz) x=vector_newsize(voi,sz*=2);
          if (flag==1.0) { x[k]=(double)qp->type; 
          } else if (flag==2.0) { x[qp->type]++; 
          } else x[k]=(double)qp->id;
        } 
        k++;
        break;
      }
    }
  }
  if (prfl && flag!=2) vector_resize(voi,k);
  _lgetconv=(double)k;
}
ENDVERBATIM
}

: INTF6[0].adjlist(List,[startid,endid,exonly])
: returns adjacency list in first arg
: startid == optional 2nd arg specifies id from which to start
: endid == optional 3rd arg specifies id to end with
: exonly == optional 4th arg specifies to only store excitatory synapse information
FUNCTION adjlist () {
  VERBATIM
  Object* pList = *hoc_objgetarg(1);
  ip=IDP; pg=ip->pg;
  int iListSz=ivoc_list_count(pList),iCell,iStartID=ifarg(2)?*getarg(2):0,\
    iEndID=ifarg(3)?*getarg(3):pg->cesz-1;
  int skipinhib = ifarg(4)?*getarg(4):0, i,j,nv,*pused=(int*)calloc(pg->cesz,sizeof(int)),iSyns=0;
  double **vvo = (double**)malloc(sizeof(double*)*iListSz),\
    *psyns=(double*)calloc(pg->cesz,sizeof(double));
  id0* rp;
  for(iCell=iStartID;iCell<=iEndID;iCell++){
    if(verbose && iCell%1000==0) printf("%d ",iCell);
    lop(pg->ce,iCell);
    if(!qp->dvt || (skipinhib && qp->inhib)){
      list_vector_resize(pList,iCell,0);
      continue;
    }
    iSyns=0;
    for(j=0;j<qp->dvt;j++){      
      rp=*((id0**) &((qp->dvi[j]->_prop->dparam)[2])); // #define sop	*_ppvar[2].pval      
      if(skipinhib && rp->inhib) continue; // if skip inhib cells...
      if(!rp->dead && qp->sprob[j]>0. && !pused[rp->id]){      
        pused[rp->id]=1;
        psyns[iSyns++]=rp->id;
      }
    }
    list_vector_resize(pList, iCell, iSyns);
    list_vector_px(pList, iCell, &vvo[iCell]);
    memcpy(vvo[iCell],psyns,sizeof(double)*iSyns);
    for(j=0;j<iSyns;j++)pused[(int)psyns[j]]=0;
  }
  free(vvo);  free(pused);  free(psyns);
  if (verbose) printf("\n");
  return 1.0;
  ENDVERBATIM
}

FUNCTION rddvi () {
  VERBATIM
  Point_process *pnnt;
  FILE* fp;
  int i, iCell;
  unsigned int iOutID;
  Object* lb;
  fp=hoc_obj_file_arg(1);
  ip=IDP; pg=ip->pg;
  printf("reading: ");
  for(iCell=0;iCell<pg->cesz;iCell++){
    if(iCell%1000==0)printf("%d ",iCell);
    lop(pg->ce,iCell);
    fread(&qp->id,sizeof(unsigned int),1,fp); // read id
    fread(&qp->type,sizeof(unsigned char),1,fp); // read type id
    fread(&qp->col,sizeof(unsigned int),1,fp); // read column id
    fread(&qp->dead,sizeof(unsigned char),1,fp); // read alive/dead status
    fread(&qp->dvt,sizeof(unsigned int),1,fp); // read divergence size
    //free up old pointers
    if(qp->del){ free(qp->del); free(qp->dvi); free(qp->sprob);
      qp->dvt=0; qp->dvi=(Point_process**)0x0; qp->del=(double*)0x0; qp->sprob=(char *)0x0; }
    //if divergence == 0 , continue
    if(!qp->dvt) continue;
    qp->dvi = (Point_process**)malloc(sizeof(Point_process*)*qp->dvt);  
    for(i=0;i<qp->dvt;i++){
      fread(&iOutID,sizeof(unsigned int),1,fp); // id of output cell
      if (!(lb=ivoc_list_item(pg->ce,iOutID))) {
        printf("INTF6:callback %d exceeds %d for list ce\n",iOutID,pg->cesz); hxe(); }
      qp->dvi[i]=(Point_process *)lb->u.this_pointer;
    }
    qp->del = (double*)malloc(sizeof(double)*qp->dvt);
    fread(qp->del,sizeof(double),qp->dvt,fp); // read divergence delays
    qp->sprob = (unsigned char*)malloc(sizeof(unsigned char)*qp->dvt);
    fread(qp->sprob,sizeof(unsigned char),qp->dvt,fp); // read divergence firing probabilities
  }
  printf("\n");
  return 1.0;
  ENDVERBATIM
}

FUNCTION svdvi () {
  VERBATIM
  Point_process *pnnt;
  FILE* fp;
  int i , iCell;
  fp=hoc_obj_file_arg(1);
  ip=IDP; pg=ip->pg;
  printf("writing: ");
  for(iCell=0;iCell<pg->cesz;iCell++){
    if(iCell%1000==0)printf("%d ",iCell);
    lop(pg->ce,iCell);
    fwrite(&qp->id,sizeof(unsigned int),1,fp); // write id
    fwrite(&qp->type,sizeof(unsigned char),1,fp); // write type id
    fwrite(&qp->col,sizeof(unsigned int),1,fp); // write column id
    fwrite(&qp->dead,sizeof(unsigned char),1,fp); // write alive/dead status
    fwrite(&qp->dvt,sizeof(unsigned int),1,fp); // write divergence size
    if(!qp->dvt)continue; //don't write empty pointers if no divergence
    for(i=0;i<qp->dvt;i++){
      pnnt=qp->dvi[i];
      fwrite(&(*(id0**)&(pnnt->_prop->dparam[2]))->id,sizeof(unsigned int),1,fp); // id of output cell
    }
    fwrite(qp->del,sizeof(double),qp->dvt,fp); // write divergence delays
    fwrite(qp->sprob,sizeof(unsigned char),qp->dvt,fp); // write divergence firing probabilities
  }
  printf("\n"); 
  return 1.0;
  ENDVERBATIM
}

: INTF6[0].setdvir(wiringlist,delaylist[,flag]) // flag default is 0 to pass to setdvi2()
: INTF6[0].setdvir(wiringlist,delaylist,startid,endid)
: INTF6[0].setdvir(wiringlist,delaylist,startid,endid,flag)
: INTF6[0].setdvir(wiringlist,delaylist,idvec,flag)
: should either use just with flag == 0 to setup all dvi outputs of cells
: or with flag == 1 to incrementally setup outputs from cells and on the last
: set of outputs from a range of cells call with flag == 2 to setup sprob and sort dvi list
: alternatively, can call setdvir with flag == 1, and at end just call INTF6.finishdvir to finalize
FUNCTION setdvir () {
  VERBATIM
  ListVec* pListWires,*pListDels;
  int i,dn,flag,dvt,idvfl,iCell,iStartID,iEndID,nidv,end; 
  double *y, *d, *idvec; unsigned char pdead;
  ip=IDP; pg=ip->pg;
  pListWires = AllocListVec(*hoc_objgetarg(1));
  idvfl=flag=0; iStartID=0; iEndID=pg->cesz-1;
  if(!pListWires){printf("setalldvi ERRA: problem initializing wires list arg!\n"); hxe();}
  pListDels = AllocListVec(*hoc_objgetarg(2));
  if(!pListDels){ printf("setalldvi ERRA: problem initializing delays list arg!\n");
    FreeListVec(&pListWires); hxe(); }
  if (ifarg(3) && !ifarg(4)) { 
    flag=(int)*getarg(3); 
  } else if (hoc_is_double_arg(3)) {
    iStartID=(int)*getarg(3);
    iEndID = (int)*getarg(4);
    if(ifarg(5)) flag=(int)*getarg(5);
  } else {
    nidv=vector_arg_px(3, &idvec);
    idvfl=1;
    if (ifarg(4)) flag=(int)*getarg(4);
  }
  end=idvfl?nidv:(iEndID-iStartID+1);
  for (i=0;i<end;i++) {
    if(i%1000==0) printf("%d",i/1000);
    iCell=idvfl?idvec[i]:(iStartID+i);
    lop(pg->ce,iCell);
    if (qp->dead) continue;
    y=pListWires->pv[i]; dvt=pListWires->plen[i];
    if(!dvt) continue; //skip empty div lists
    d=pListDels->pv[i];  dn=pListDels->plen[i];
    if (dn!=dvt) {printf("setdvir() ERR vec sizes for wire,delay list entries not equal %d: %d %d\n",i,dvt,dn); hxe();}
    setdvi2(y,d,0x0,dvt,flag,0x0,0x0);
  }
  FreeListVec(&pListWires);
  FreeListVec(&pListDels);
  return 1.0;
  ENDVERBATIM
}

PROCEDURE clrdvi () {
  VERBATIM
  int i;
  ip=IDP; pg=ip->pg;
  for (i=0;i<pg->cesz;i++) { 
    lop(pg->ce,i);
    if (qp->dvt!=0x0) {
      free(qp->dvi); free(qp->del); free(qp->sprob);
      qp->dvt=0; qp->dvi=(Point_process**)0x0; qp->del=(double*)0x0; qp->sprob=(char *)0x0;
      if(wsetting==1) freesywv(qp);
    }
  }
  ENDVERBATIM
}

: int.setdviv(prevec,postvec,delvec,distal,wt1,wt2)
PROCEDURE setdviv () {
  VERBATIM
  int i,j,k,l,nprv,dvt,*scr; double *prv,*pov,*dlv,x,*ds,*w1,*w2; char* s;
  ip=IDP; pg=ip->pg;
  nprv=vector_arg_px(1, &prv);
  i=vector_arg_px(2, &pov);
  j=vector_arg_px(3, &dlv);
  if(ifarg(4)) { s=(char*)calloc((l=vector_arg_px(4,&ds)),sizeof(char)); for(k=0;k<l;k++) s[k]=ds[k]; k=0;
  } else s=0x0;
  if (nprv!=i || i!=j || j!=l) {printf("intf:setdviv ERRA: %d %d %d %d\n",nprv,i,j,l); hxe();}
  if (wsetting==1) {
    i=vector_arg_px(5, &w1);
    j=vector_arg_px(6, &w2);
    if (nprv!=i || i!=j) {printf("intf:setdviv ERRB: %d %d %d\n",nprv,i,j); hxe();}
  }
  // start by counting the prids so will know the size that we need for realloc()
  scr=(int *)ecalloc(pg->cesz, sizeof(int));
  for (i=0;i<pg->cesz;i++) scr[i]=0;
  for (i=0,j=-1;i<nprv;i++) {
    if ((int)prv[i]<j) { printf("intf:setdviv ERRC vecs should be sorted by prid vec\n");hxe(); }
    j=(int)prv[i];
    scr[j]++;
  }
  if (ip->dbx>1) for (i=0;i<pg->cesz;i++) printf("%d ",scr[i]);
  for (i=-1,k=0;k<nprv;k+=dvt) { if(i%1000==0) printf(".");
    if ((int)prv[k]==i) {printf("intf:setdviv ERRD number repeated %g %d %d\n",prv[k],i,k);hxe();}
    i=(int)prv[k]; // index for presyn cell
    lop(pg->ce,i); // set the container to that cell
    dvt=scr[i]; // the number of postsyns for that
    if (ip->dbx>0) printf("DBA:%d,%d,%d ",i,dvt,k);
    if (qp->dead) continue;
    if (dvt>0) {
      if (wsetting==1) {
        setdvi3(pov+k,dlv+k,s+k,dvt,w1+k,w2+k); // no flag -- will just replace the divergence list
      } else {
        setdvi2(pov+k,dlv+k,s?s+k:0x0,dvt,1,0x0,0x0);
      }
    }
  }
  if(s) free(s);
  ENDVERBATIM
}

VERBATIM
void setupsywv (id0* p, int sz) {
  p->syw1 = p->syw1!=0x0 ? (double*) realloc((double*) p->syw1, sz*sizeof(double)) :
                           (double*) calloc(sz,sizeof(double));

  p->syw2 = p->syw2!=0x0 ? (double*) realloc((double*) p->syw2, sz*sizeof(double)) :
                           (double*) calloc(sz,sizeof(double));
}

//void myfree(void** p) {
//  int* ip;
//  if(p[0]) free(p[0]);
//  ip = (int*) p[0];
//  ip = 0x0;
//}

void freesywv (id0* p) {
  if(p->syw1) free(p->syw1); p->syw1=0x0;
  if(p->syw2) free(p->syw2); p->syw2=0x0;
}
ENDVERBATIM

: intf.setsywv(weight vector 1, weight vector 2)
FUNCTION setsywv () {
  VERBATIM
  int sz,n1,n2; double *psyw1,*psyw2; id0* ip;
  ip=IDP; pg=ip->pg; sz=ip->dvt;
  if((n1=vector_arg_px(1, &psyw1))!=sz || (n2=vector_arg_px(2, &psyw2))!=sz) {
    printf("setsywv ERRA: make sure weight vector sizes (%d,%d) same size as div(%d)\n",n1,n2,sz);
    return 0.0;
  }
  setupsywv(ip,sz); // setup pointers
  memcpy(ip->syw1,psyw1,sizeof(double)*sz); // copy
  memcpy(ip->syw2,psyw2,sizeof(double)*sz);
  return sz;
  ENDVERBATIM
}

: intf.getsywv(weight vector 1, weight vector 2)
FUNCTION getsywv () {
  VERBATIM
  int sz,n1,n2; double *psyw1,*psyw2; id0* ip;
  ip=IDP; pg=ip->pg; sz=ip->dvt;
  if(!ip->syw1 || !ip->syw2) {
    printf("getsywv ERRA: syw1,syw2 were never initialized with setsywv!\n");
    return 0.0;
  }
  if((n1=vector_arg_px(1, &psyw1))!=sz || (n2=vector_arg_px(2, &psyw2))!=sz) {
    printf("getsywv ERRB: make sure weight vector sizes (%d,%d) same size as div(%d)\n",n1,n2,sz);
    return 0.0;
  }
  memcpy(psyw1,ip->syw1,sizeof(double)*sz); // copy
  memcpy(psyw2,ip->syw2,sizeof(double)*sz);
  return sz;
  ENDVERBATIM
}

VERBATIM
// get presynaptic excitatory cells in a double*, psz[0] has size
int* getpeconv (id0* ip,int* psz) {
  Point_process **das; int* pfrom;
  int i,j,k,dvt;
  *psz=ip->dvt>0?ip->dvt:16; pg=ip->pg;
  pfrom=(int*) calloc(psz[0],sizeof(int));
  for (i=0,k=0; i<pg->cesz; i++) {
    lop(pg->ce,i);
    if(qp->inhib) continue; // skip presynaptic inhib cells
    dvt=qp->dvt;
    das=qp->dvi;
    for (j=0;j<dvt;j++) {
      if (ip==*((id0**) &((das[j]->_prop->dparam)[2]))) {
        if (k>=*psz) {
          psz[0]*=2;
          pfrom=(int*) realloc((void*)pfrom,psz[0]*sizeof(int));
        }
        pfrom[k]=qp->id;
        k++;
        break;
      }
    }
  }
  *psz=k;
  return pfrom;
}

// get presynaptic inhibitory cells in a double*, psz[0] has size
int* getpiconv (id0* ip,int* psz) {
  Point_process **das; int* pfrom;
  int i,j,k,dvt;
  *psz=ip->dvt>0?ip->dvt:16; pg=ip->pg;
  pfrom=(int*) calloc(psz[0],sizeof(int));
  for (i=0,k=0; i<pg->cesz; i++) {
    lop(pg->ce,i);
    if(!qp->inhib) continue; // skip presynaptic excitatory cells
    dvt=qp->dvt;
    das=qp->dvi;
    for (j=0;j<dvt;j++) {
      if (ip==*((id0**) &((das[j]->_prop->dparam)[2]))) {
        if (k>=*psz) {
          psz[0]*=2;
          pfrom=(int*) realloc((void*)pfrom,psz[0]*sizeof(int));
        }
        pfrom[k]=qp->id;
        k++;
        break;
      }
    }
  }
  *psz=k;
  return pfrom;
}


int myfindidx (id0* ppre,int poid) {
  int i; Point_process** das; id0* ppo;
  das=ppre->dvi;
  for(i=0;i<ppre->dvt;i++) {
    ppo=*((id0**) &((das[i]->_prop->dparam)[2])); // #define sop	*_ppvar[2].pval
    if(ppo->id==poid) return i;
  }
  return -1;
}

// apply dopamine eligibility trace from E->X cells
// pcell is a cell that just spiked, myspkt is time of spike
static void applyEDOPE (id0* pcell,double myspkt) {
  int poid,prid,sz,i,idx; postgrp* pg; double d,inc,tmp,pinc,tau,maxw; id0* ppre, *ppo;
  if(seadsetting!=3.) return; // seadsetting==3 for EDOPE, must be set before network setup
  poid=pcell->id; pg=pcell->pg;
  if(pcell->dbx<-1) printf("applyEDOPE: pcell=%p\n",pcell);
  if (FORWELIGTR) {  // if forward eligibility traces are turned on (post after pre)
    for(i=0;i<pcell->econvsz;i++) {//check presynaptic E cells, if they fired within   maxplastt turn on eligibility trace
      prid = pcell->peconv[i];                // presynaptic id
      if(pg->lastspk[prid]<0) continue;     // cell didn't spike
      if( (d = myspkt - pg->lastspk[prid] ) > maxplastt) continue;  // time difference
      if(verbose>2) printf("spk%d:%g, spk%d:%g, d=%g\n",prid,pg->lastspk[prid],poid,pg->lastspk[poid],d);
      ppre = getlp(pg->ce,prid);            // get pointer to presynaptic cell
      idx = myfindidx(ppre,poid);           // find the index of poid in ppre's div
      if(idx<0){printf("**** applyEDOPE ERR: bad idx = %d!!!!!!!!!\n",idx); return;}
      if( ! ( inc = ppre->pplastinc[idx] ) ) continue; 
      ppre->pdope[idx] = t; // store time elig. trace turned on
      if(verbose>2) printf("EDOPEA:ppre->inhib=%d,pcell->inhib=%d,d=%g,tau=%g,d/tau=%g, %d->%d\n",ppre->inhib,pcell->inhib,d,tau,d/tau,prid,poid);
    }
  }
  if (BACKELIGTR) { // if backward eligibility traces are turned on (pre after post)
    if(pcell->inhib) return; // only EDOPE from E cells
    for(i=0;i<pcell->dvt;i++) { // check postsynaptic targets, if they fired within maxplastt, turn on eligibility trace
      ppo=*((id0**) &((pcell->dvi[i]->_prop->dparam)[2])); // #define sop	*_ppvar[2].pval
      poid = ppo->id;
      if(pg->lastspk[poid]<0) continue;
      if( (d = myspkt - pg->lastspk[poid] ) <= maxplastt) {
        if( ! ( inc = pcell->pplastinc[i] ) ) continue;
        pcell->pdope[i] = -t; // store time elig. trace turned on, -t means it was post-before-pre
        if(verbose>2) printf("EDOPEB:ppo->inhib=%d,d=%g,tau=%g,d/tau=%g, %d->%d\n",ppo->inhib,d,tau,d/tau,prid,poid);
      }    
    } 
  }
}

// apply dopamine eligibility trace from I->X cells
// pcell is a cell that just spiked, myspkt is time of spike
// GLC, 1/12/12 -- It's not really clear to me how eligibility traces should be 
// implemented in the case of I->X connections.  Until we've done more literature 
// search on this, I think we should avoid using DA learning on I->X connections.
static void applyIDOPE (id0* pcell,double myspkt) {
  int poid,prid,sz,i,idx; postgrp* pg; double d,inc,tmp,pinc,tau,maxw; id0* ppre, *ppo;
  if(seadsetting!=3.) return; // seadsetting==3 for IDOPE, must be set before network setup
  poid=pcell->id; pg=pcell->pg;
  if(pcell->dbx<-1) printf("applyplast: pcell=%p\n",pcell);
  if (FORWELIGTR) {  // if forward eligibility traces are turned on (post after pre)
    for(i=0;i<pcell->iconvsz;i++) {//check presynaptic I cells, if they fired earlier, depress synapse
      prid = pcell->piconv[i];                // presynaptic id
      if(pg->lastspk[prid]<0) continue;     // cell didn't spike
      if( (d = myspkt - pg->lastspk[prid] ) > maxplastt) continue;  // time difference
      if(verbose>2) printf("spk%d:%g, spk%d:%g, d=%g\n",prid,pg->lastspk[prid],poid,pg->lastspk[poid],d);
      ppre = getlp(pg->ce,prid);            // get pointer to presynaptic cell
      idx = myfindidx(ppre,poid);           // find the index of poid in ppre's div
      if(idx<0){printf("**** applyISSTDP ERR: bad idx = %d!!!!!!!!!\n",idx); return;}
      if( ! ( inc = ppre->pplastinc[idx] ) ) continue; 
      ppre->pdope[idx] = t; // store time elig. trace turned on
      if(verbose>2) printf("IDOPEA:ppre->inhib=%d,pcell->inhib=%d,d=%g,tau=%g,d/tau=%g, %d->%d\n",ppre->inhib,pcell->inhib,d,tau,d/tau,prid,poid);
    }
  }
  if(BACKELIGTR) { // if backward eligibility traces are turned on (pre after post)
    if(!pcell->inhib) return; // IDOPE only from I cells
    for(i=0;i<pcell->dvt;i++) { // check postsynaptic targets, if within maxplastt, 
      ppo=*((id0**) &((pcell->dvi[i]->_prop->dparam)[2])); // #define sop	*_ppvar[2].pval
      poid = ppo->id;
      if(pg->lastspk[poid]<0) continue;
      if( (d = myspkt - pg->lastspk[poid] ) <= maxplastt) {
        if( ! ( inc = pcell->pplastinc[i] ) ) continue;
        pcell->pdope[i] = -t; // store time elig. trace turned on, -t means it was post-before-pre
        if(verbose>2) printf("IDOPEB:ppo->inhib=%d,d=%g,tau=%g,d/tau=%g, %d->%d\n",ppo->inhib,d,tau,d/tau,prid,poid);
      }    
    }
  }
}

// apply plasticity from E->X cells
// pcell is a cell that just spiked, myspkt is time of spike
static void applyEXSTDP (id0* pcell,double myspkt) {
  int poid,prid,sz,i,idx; postgrp* pg; double d,inc,tmp,pinc,tau,maxw; id0* ppre, *ppo;
  if(seadsetting!=3.) return; // seadsetting==3 for STDP, must be set before network setup
  poid=pcell->id; pg=pcell->pg;
  if(pcell->dbx<-1) printf("applyEXSTDP: pcell=%p\n",pcell);
  for(i=0;i<pcell->econvsz;i++) {//check presynaptic E cells, if they fired earlier, potentiate synapse
    prid = pcell->peconv[i];                // presynaptic id
    if(pg->lastspk[prid]<0) continue;     // cell didn't spike
    if( (d = myspkt - pg->lastspk[prid] ) > maxplastt) continue;  // time difference
    if(verbose>2) printf("spk%d:%g, spk%d:%g, d=%g\n",prid,pg->lastspk[prid],poid,pg->lastspk[poid],d);
    ppre = getlp(pg->ce,prid);            // get pointer to presynaptic cell
    idx = myfindidx(ppre,poid);           // find the index of poid in ppre's div
    if(idx<0){printf("**** applyEXSTDP ERR: bad idx = %d!!!!!!!!!\n",idx); return;}
    if( ! ( inc = ppre->pplastinc[idx] ) ) continue; 
    tau = ppre->pplasttau[idx];
    maxw = ppre->pplastmaxw[idx];
    tmp = ppre->wgain[idx]; // temp - holds original wgain level  
    if(SOFTSTDP) inc *= (1.0 - tmp / maxw); // soft bound for potentiation
    ppre->wgain[idx] += EPOTW * inc * exp( -d / tau ); // increment the wgain of the synapse
    if(ppre->wgain[idx]<0.) ppre->wgain[idx]=0.; // check bounds of wgain
    else if(!SOFTSTDP && ppre->wgain[idx]>maxw) ppre->wgain[idx]=maxw;
    if(verbose>2) printf("PLAST:ppre->inhib=%d,pcell->inhib=%d,d=%g,tau=%g,d/tau=%g, %d->%d: inc=%g, wgA=%g, wgB=%g\n",ppre->inhib,pcell->inhib,d,tau,d/tau,prid,poid,inc,tmp,ppre->wgain[idx]);
  }
  if(pcell->inhib) return; // only STDP from E cells
  for(i=0;i<pcell->dvt;i++) { // check postsynaptic targets, if they fired earlier, depress the synapse
    ppo=*((id0**) &((pcell->dvi[i]->_prop->dparam)[2])); // #define sop	*_ppvar[2].pval
    poid = ppo->id;
    if(pg->lastspk[poid]<0) continue;
    if( (d = myspkt - pg->lastspk[poid] ) < maxplastt) {
      if( ! ( inc = pcell->pplastinc[i] ) ) continue;
      tau = pcell->pplasttau[i];
      maxw = pcell->pplastmaxw[i];
      tmp = pcell->wgain[i]; // temp - holds original wgain level  
      if(SOFTSTDP) inc *= (tmp / maxw); // soft bound for depression
      pcell->wgain[i] -= EDEPW * inc * exp( -d / tau ); // increment the wgain of the synapse
      if(pcell->wgain[i]<0.) pcell->wgain[i]=0.; // check bounds of wgain
      else if(!SOFTSTDP && pcell->wgain[i]>maxw) pcell->wgain[i]=maxw;
      if(verbose>2) printf("DEP:ppo->inhib=%d,d=%g,tau=%g,d/tau=%g, %d->%d: inc=%g, wgA=%g, wgB=%g\n",ppo->inhib,d,tau,d/tau,prid,poid,inc,tmp,pcell->wgain[i]);
    }    
  }
}

// apply plasticity from I->X cells
// pcell is a cell that just spiked, myspkt is time of spike
static void applyIXSTDP (id0* pcell,double myspkt) {
  int poid,prid,sz,i,idx; postgrp* pg; double d,inc,tmp,pinc,tau,maxw; id0* ppre, *ppo;
  if(seadsetting!=3.) return; // seadsetting==3 for STDP, must be set before network setup
  poid=pcell->id; pg=pcell->pg;
  if(pcell->dbx<-1) printf("applyplast: pcell=%p\n",pcell);
  for(i=0;i<pcell->iconvsz;i++) {//check presynaptic I cells, if they fired earlier, depress synapse
    prid = pcell->piconv[i];                // presynaptic id
    if(pg->lastspk[prid]<0) continue;     // cell didn't spike
    if( (d = myspkt - pg->lastspk[prid] ) > maxplastt) continue;  // time difference
    if(verbose>2) printf("spk%d:%g, spk%d:%g, d=%g\n",prid,pg->lastspk[prid],poid,pg->lastspk[poid],d);
    ppre = getlp(pg->ce,prid);            // get pointer to presynaptic cell
    idx = myfindidx(ppre,poid);           // find the index of poid in ppre's div
    if(idx<0){printf("**** applyISSTDP ERR: bad idx = %d!!!!!!!!!\n",idx); return;}
    if( ! ( inc = ppre->pplastinc[idx] ) ) continue; 
    tau = ppre->pplasttau[idx];
    maxw = ppre->pplastmaxw[idx];
    tmp = ppre->wgain[idx]; // temp - holds original wgain level  
    if(SOFTSTDP) inc *= (tmp / maxw); // soft bound for depression
    ppre->wgain[idx] -= IDEPW * inc * exp( -d / tau ); // increment the wgain of the synapse
    if(ppre->wgain[idx]<0.) ppre->wgain[idx]=0.; // check bounds of wgain
    else if(!SOFTSTDP && ppre->wgain[idx]>maxw) ppre->wgain[idx]=maxw;
    if(verbose>2) printf("DEP:ppre->inhib=%d,pcell->inhib=%d,d=%g,tau=%g,d/tau=%g, %d->%d: inc=%g, wgA=%g, wgB=%g\n",ppre->inhib,pcell->inhib,d,tau,d/tau,prid,poid,inc,tmp,ppre->wgain[idx]);
  }
  if(!pcell->inhib) return; // this STDP only from I cells
  for(i=0;i<pcell->dvt;i++) { // check postsynaptic targets, if they fired earlier, potentiate the synapse
    ppo=*((id0**) &((pcell->dvi[i]->_prop->dparam)[2])); // #define sop	*_ppvar[2].pval
    poid = ppo->id;
    if(pg->lastspk[poid]<0) continue;
    if( (d = myspkt - pg->lastspk[poid] ) < maxplastt) {
      if( ! ( inc = pcell->pplastinc[i] ) ) continue;
      tau = pcell->pplasttau[i];
      maxw = pcell->pplastmaxw[i];
      tmp = pcell->wgain[i]; // temp - holds original wgain level  
      if(SOFTSTDP) inc *= (1.0 - tmp / maxw); // soft bound for potentiation
      pcell->wgain[i] += IPOTW * inc * exp( -d / tau ); // increment the wgain of the synapse
      if(pcell->wgain[i]<0.) pcell->wgain[i]=0.; // check bounds of wgain
      else if(!SOFTSTDP && pcell->wgain[i]>maxw) pcell->wgain[i]=maxw;
      if(verbose>2) printf("PLAST:ppo->inhib=%d,d=%g,tau=%g,d/tau=%g, %d->%d: inc=%g, wgA=%g, wgB=%g\n",ppo->inhib,d,tau,d/tau,prid,poid,inc,tmp,pcell->wgain[i]);
    }    
  }
}

ENDVERBATIM

: intf.geteconv(vec) - get presynaptic E cell IDs
FUNCTION geteconv () {
  VERBATIM
  int i; double *x; void *voi;
  ip=IDP; pg=ip->pg;
  if(!ip->peconv) ip->peconv=getpeconv(ip,&ip->econvsz);
  voi=vector_arg(1);
  x=vector_newsize(voi,ip->econvsz);
  for(i=0;i<ip->econvsz;i++) x[i]=(double)ip->peconv[i];
  return ip->econvsz;
  ENDVERBATIM
}

: intf.geticonv(vec) - get presynaptic I cell IDs
FUNCTION geticonv () {
  VERBATIM
  int i; double *x; void *voi;
  ip=IDP; pg=ip->pg;
  if(!ip->piconv) ip->piconv=getpiconv(ip,&ip->iconvsz);
  voi=vector_arg(1);
  x=vector_newsize(voi,ip->iconvsz);
  for(i=0;i<ip->iconvsz;i++) x[i]=(double)ip->piconv[i];
  return ip->iconvsz;
  ENDVERBATIM
}

: finishdvi2 () -- finalize dvi , sort dvi , allocate and set sprob
VERBATIM
static int finishdvi2 (struct ID0* p) {
  Point_process **da,**das;
  double *db,*dbs,*w1,*w1s,*w2,*w2s;
  char *syns,*synss;
  int i, dvt;
  db=p->del;
  da=p->dvi; 
  dvt=p->dvt;
  syns=p->syns;
  dbs=(double*)malloc(dvt*sizeof(double)); // sorted delays
  das=(Point_process**)malloc(dvt*sizeof(Point_process*)); // parallel sorted dvi
  synss=(char*)malloc(dvt*sizeof(char)); // sorted syns
  if(wsetting==1 && p->syw1 && p->syw2) {
    w1=p->syw1;
    w2=p->syw2;
    w1s=(double*)malloc(dvt*sizeof(double)); //mem for sorted weights
    w2s=(double*)malloc(dvt*sizeof(double));
    gsort5(db,da,syns,w1,w2,dvt,dbs,das,synss,w1s,w2s); //sort it all
    p->syw1=w1s; p->syw2=w2s; //sorted weights
    free(w1); free(w2); // free old ones
  } else gsort3(db,da,syns,dvt,dbs,das,synss);
  p->del=dbs; p->dvi=das; p->syns=synss;// sorted versions
  free(db); free(da); free(syns); // free old mem
  p->sprob=(unsigned char*)realloc((void*)p->sprob,(size_t)dvt*sizeof(char));// release probability
  for (i=0;i<dvt;i++) p->sprob[i]=1; // start out with all firing
  p->wgain=(double*)realloc((void*)p->wgain,(size_t)dvt*sizeof(double));//synaptic weight gain
  for (i=0;i<dvt;i++) p->wgain[i]=1.0; // start out at wmat level
  p->peconv = getpeconv(p,&p->econvsz); // get econv
  p->piconv = getpiconv(p,&p->iconvsz); // get iconv
  if(seadsetting==3) {
    p->pplasttau = (double*)realloc((void*)p->pplasttau,(size_t)dvt*sizeof(double));
    p->pplastinc = (double*)realloc((void*)p->pplastinc,(size_t)dvt*sizeof(double));
    p->pplastmaxw = (double*)realloc((void*)p->pplastmaxw,(size_t)dvt*sizeof(double));
    if(DOPE) p->pdope = (double*)realloc((void*)p->pdope,(size_t)dvt*sizeof(double));
  }
}
ENDVERBATIM

: finalize dvi for all cells
PROCEDURE finishdvir () {
  VERBATIM
  int iCell;
  ip=IDP; pg=ip->pg;
  for(iCell=0;iCell<pg->cesz;iCell++){
    lop(pg->ce,iCell);
    finishdvi2(qp);
  }
  ENDVERBATIM
}

: finishdvi() -- finalize dvi , sort dvi, allocate and set sprob, for this single cell
PROCEDURE finishdvi () {
VERBATIM
  finishdvi2(IDP);
ENDVERBATIM
}

: intf.setplast(vwgain,vplasttau,vplastinc,vplastmaxw)
: seadsetting must be 3, vectors must have same size as dvi
FUNCTION setplast () {
  VERBATIM
  double *wgain,*pplasttau,*pplastinc,*pplastmaxw;
  if(seadsetting!=3) {printf("setplast ERR0: seadsetting must be 3, plast mode off!\n"); return 0;}
  ip=IDP; pg=ip->pg;
  if(vector_arg_px(1,&wgain) != ip->dvt ||
     vector_arg_px(2,&pplasttau) != ip->dvt ||
     vector_arg_px(3,&pplastinc) != ip->dvt ||
     vector_arg_px(4,&pplastmaxw) != ip->dvt) {printf("setplast ERR1: input vectors must have size %d!\n",ip->dvt); return 0;}
  memcpy(ip->wgain,wgain,sizeof(double)*ip->dvt);
  memcpy(ip->pplasttau,pplasttau,sizeof(double)*ip->dvt);
  memcpy(ip->pplastinc,pplastinc,sizeof(double)*ip->dvt);
  memcpy(ip->pplastmaxw,pplastmaxw,sizeof(double)*ip->dvt);
  return 0.0;
  ENDVERBATIM
}

: intf.getplast(vwgain,vplasttau,vplastinc,vplastmaxw)
: seadsetting must be 3, vectors must have same size as dvi
FUNCTION getplast () {
  VERBATIM
  double *wgain,*pplasttau,*pplastinc,*pplastmaxw;
  if(seadsetting!=3) {printf("getplast ERR0: seadsetting must be 3, plast mode off!\n"); return 0;}
  ip=IDP; pg=ip->pg;
  ip=IDP; pg=ip->pg;
  if(vector_arg_px(1,&wgain) != ip->dvt ||
     vector_arg_px(2,&pplasttau) != ip->dvt ||
     vector_arg_px(3,&pplastinc) != ip->dvt ||
     vector_arg_px(4,&pplastmaxw) != ip->dvt) {printf("getplast ERR1: output vectors must have size %d!\n",ip->dvt); return 0;}
  memcpy(wgain,ip->wgain,sizeof(double)*ip->dvt);
  memcpy(pplasttau,ip->pplasttau,sizeof(double)*ip->dvt);
  memcpy(pplastinc,ip->pplastinc,sizeof(double)*ip->dvt);
  memcpy(pplastmaxw,ip->pplastmaxw,sizeof(double)*ip->dvt);
  return 1.0;
  ENDVERBATIM
}

: setdvi(cell#s,dels[,dist,flag,w1,w2]) flag 1: grow internal vecs; flag 2: grow and do final sort
: w1,w2 are weight vectors -- i.e. for AM2,NM2. should only be used when wsetting==1
PROCEDURE setdvi () {
VERBATIM {
  int i,j,k,dvt,flag; double *d, *y, *ds, *w1, *w2; char* s;
  if (! ifarg(1)) {printf("setdvi(v1,v2[,v3,flag]): v1:cell#s; v2:delays; v3:distal synapses\n"); return 0; }
  ip=IDP; pg=ip->pg; // this should only be called after jitcondiv()
  if (ip->dead) return 0;
  dvt=vector_arg_px(1, &y);
  i=vector_arg_px(2, &d);
  s=ifarg(3)?(char*)calloc((j=vector_arg_px(3,&ds)),sizeof(char)):0x0;
  if(s) for(k=0;k<j;k++) s[k]=(char)ds[k];
  if (ifarg(4)) flag=(int)*getarg(4); else flag=0;
  if (i!=dvt || i==0 || (j>0 && j!=i)) {printf("setdvi() ERR vec sizes: %d %d %d\n",dvt,i,j); hxe();}
  w1=w2=0x0;
  if(ifarg(5) && wsetting!=1){printf("setdvi ERR: only use weight vecs when wsetting==1!\n"); hxe();}
  if(ifarg(5) && dvt!=vector_arg_px(5,&w1)){printf("setdvi ERR: wrong size for w1 vector!\n"); hxe();}
  if(ifarg(6) && dvt!=vector_arg_px(6,&w2)){printf("setdvi ERR: wrong size for w2 vector!\n"); hxe();}
  setdvi2(y,d,s,dvt,flag,w1,w2);
  }
ENDVERBATIM
}

VERBATIM
// setdvi2(divid_vec,del_vec,syns_vec,div_cnt,flag,w1,w2)
// flag 1 means just augment, 0or2: sort by del, 0: clear lists and replace
static int setdvi2 (double *y,double *d,char* s,int dvt,int flag,double* w1,double* w2) {
  int i,j,ddvi; double *db, *dbs, *w1s, *w2s; unsigned char pdead; unsigned int b,e; char* syns;
  Object *lb; Point_process *pnnt, **da, **das;
  ddvi=(int)DEAD_DIV;
  ip=IDP; pg=ip->pg;
  if(wsetting==1 && (!w1 || !w2)) {
    printf("setdvi2 ERR: wsetting==1, must provide w1,w2 arrays!\n");
    hxe();
  }
  if (flag==0) { b=0; e=dvt; // begin to end
    if (ip->dvi) { 
      free(ip->dvi); free(ip->del); free(ip->sprob); free(ip->syns); 
      ip->dvt=0; ip->dvi=(Point_process**)0x0; ip->del=(double*)0x0; ip->sprob=(char *)0x0; ip->syns=(char*)0x0;
      if(ip->wgain){free(ip->wgain); ip->wgain=0x0;}
      if(ip->peconv){free(ip->peconv); ip->peconv=0x0;}
      if(ip->piconv){free(ip->piconv); ip->piconv=0x0;}
      if(ip->pplasttau){free(ip->pplasttau);ip->pplasttau=0x0;}
      if(ip->pplastinc){free(ip->pplastinc);ip->pplastinc=0x0;}
      if(ip->pplastmaxw){free(ip->pplastmaxw);ip->pplastmaxw=0x0;}
      if(ip->pdope){free(ip->pdope);ip->pdope=0x0;}
      if(wsetting==1) freesywv(ip);
    } // make sure all null pointers for realloc
  } else { 
    if (ip->dvt==0) {
      ip->dvi=(Point_process**)0x0; ip->del=(double*)0x0; ip->sprob=(char *)0x0; ip->syns=(char*)0x0;
      ip->wgain=0x0; ip->peconv=0x0; ip->piconv=0x0;
      ip->pplasttau=0x0; ip->pplastinc=0x0; ip->pplastmaxw=0x0; ip->pdope=0x0;
      if(wsetting==1) freesywv(ip);
    }
    b=ip->dvt; 
    e=ip->dvt+dvt; // dvt is amount to grow
  }
  da=(Point_process **)realloc((void*)ip->dvi,(size_t)(e*sizeof(Point_process *)));
  db=(double*)realloc((void*)ip->del,(size_t)(e*sizeof(double)));
  syns=(char*)realloc((void*)ip->syns,(size_t)(e*sizeof(char)));  
  if(wsetting==1) {
    w1s=(double*)realloc((void*)ip->syw1,(size_t)(e*sizeof(double)));
    w2s=(double*)realloc((void*)ip->syw2,(size_t)(e*sizeof(double)));
  }
  for (i=b,j=0;j<dvt;j++) { // i thru da[] j thru y, k to append
    // div can grow at lower rate if dead cells are encountered
    if (!(lb=ivoc_list_item(pg->ce,(unsigned int)y[j]))) {
      printf("INTF6:callback %g exceeds %d for list ce\n",y[j],pg->cesz); hxe(); }
      pnnt=(Point_process *)lb->u.this_pointer;
      if (ddvi==1 || !(pdead=(*(id0**)&(pnnt->_prop->dparam[2]))->dead)) {
        da[i]=pnnt; db[i]=d[j]; syns[i]=s?s[j]:0; 
        if(wsetting==1){w1s[i]=w1[j]; w2s[i]=w2[j];}
        i++;
      }
  }
  if ((dvt=i)<e) { // will need to shrink these arrays
    da=(Point_process **)realloc((void*)da,(size_t)(dvt*sizeof(Point_process *)));
    db=(double*)realloc((void*)db,(size_t)(dvt*sizeof(double)));
    syns=(char*)realloc((void*)syns,(size_t)(dvt*sizeof(char)));
    if(wsetting==1) {
      w1s=(double*)realloc((void*)w1s,(size_t)(dvt*sizeof(double)));
      w2s=(double*)realloc((void*)w2s,(size_t)(dvt*sizeof(double)));
    }
  }
  ip->dvt=dvt; ip->del=db; ip->dvi=da; ip->syns=syns;
  if(wsetting==1){ip->syw1=w1s; ip->syw2=w2s;}
  if (flag!=1) finishdvi2(ip); // do sort
}
ENDVERBATIM

VERBATIM
// setdvi3(divid_vec,del_vec,syns_vec,div_cnt,w1,w2)
// based on setdvi2() but uses qp statt IDP and does reallocs
static int setdvi3 (double *y, double *d, char* s, int dvt, double* w1, double* w2) {
  int i,j,ddvi; double *db, *dbs, *w1s, *w2s; unsigned char pdead; unsigned int b,e; char* syns;
  Object *lb; Point_process *pnnt, **da, **das;
  ddvi=(int)DEAD_DIV;
  ip=qp; pg=ip->pg;
  e=dvt; // begin to end
  da=(Point_process **)realloc((void*)ip->dvi,(size_t)(e*sizeof(Point_process *)));
  db=(double*)realloc((void*)ip->del,(size_t)(e*sizeof(double)));
  syns=(char*)realloc((void*)ip->syns,(size_t)(e*sizeof(char)));  
  w1s=(double*)realloc((void*)ip->syw1,(size_t)(e*sizeof(double)));
  w2s=(double*)realloc((void*)ip->syw2,(size_t)(e*sizeof(double)));
  for (i=0,j=0;j<dvt;i++,j++) { // i thru da[] j thru y, k to append
    // div can grow at lower rate if dead cells are encountered
    if (!(lb=ivoc_list_item(pg->ce,(unsigned int)y[j]))) {
      printf("INTF6:callback %g exceeds %d for list ce\n",y[j],pg->cesz); hxe(); }
      pnnt=(Point_process *)lb->u.this_pointer;
      if (ddvi==1 || !(pdead=(*(id0**)&(pnnt->_prop->dparam[2]))->dead)) {
        da[i]=pnnt; db[i]=d[j]; syns[i]=s?s[j]:0; 
        w1s[i]=w1[j]; w2s[i]=w2[j];
      }
  }
  ip->dvt=dvt; ip->del=db; ip->dvi=da; ip->syns=syns;
  ip->syw1=w1s; ip->syw2=w2s;
  finishdvi2(ip); // do sort
}
ENDVERBATIM

: prune(p[,potype,rand_seed]) // prune synapses with prob p [0,1], ie 0.1 prunes 10% of the divergence
: prune(vec) // fill in the pruning vec with binary values from vec
PROCEDURE prune () {
  VERBATIM 
  {
  id0* ppost; double *x, p; int nx,j,potype;
  ip=IDP; pg=ip->pg;
  if (hoc_is_double_arg(1)) { // prune a certain percent of targets
    p=*getarg(1);
    if (p<0 || p>1) {printf("INTF6:pruneERR0:need # [0,1] to prune [ALL,NONE]: %g\n",p); hxe();}
    if (p==1.) printf("INTF6pruneWARNING: pruning 100% of cell %d\n",ip->id);
    if (verbose && ip->dvt>dscrsz) {
      printf("INTF6pruneB:Div exceeds dscrsz: %d>%d\n",ip->dvt,dscrsz); hxe(); }
    if (p==0.) {
      for (j=0;j<ip->dvt;j++) ip->sprob[j]=1; // unprune completely
      return 0; // now that unpruning is done, can return
    }
    potype=ifarg(2)?(int)*getarg(2):-1;
    sead=(ifarg(3))?(unsigned int)*getarg(3):GetDVIDSeedVal(ip->id);//seed for divergence and delays
    mcell_ran4(&sead, dscr , ip->dvt, 1.0); // random var (0,1)
    if(potype==-1){ // prune all types of synapses
      for (j=0;j<ip->dvt;j++) if (dscr[j]<p) ip->sprob[j]=0; // prune with prob p
    } else { // only prune synapses with postsynaptic type == potype
      for (j=0;j<ip->dvt;j++){
        ppost=*((id0**) &((ip->dvi[j]->_prop->dparam)[2])); // #define sop *_ppvar[2].pval
        if (ppost->type==potype && dscr[j]<p) ip->sprob[j]=0; // prune with prob p
      }
    }
  } else { // confusing arg1==0->sprob[j]=1 for all j; but arg1=[0] (a vector)->sprob[0]=0 
    if (verbose) printf("INTF6 WARNING prune(vec) deprecated: use intf.sprob(vec) instead\n");
    nx=vector_arg_px(1,&x);
    if (nx!=ip->dvt) {printf("INTF6:pruneERRA:Wrong size vector:%d!=%d\n",nx,ip->dvt); hxe();}
    for (j=0;j<ip->dvt;j++) ip->sprob[j]=(unsigned char)x[j];
  }
  }
ENDVERBATIM
}

PROCEDURE sprob () {
  VERBATIM 
  {
  double *x; int nx,j;
  ip=IDP; pg=ip->pg;
  nx=vector_arg_px(1,&x);
  if (nx!=ip->dvt) {printf("INTF6:pruneERRA:Wrong size vector:%d!=%d\n",nx,ip->dvt); hxe();}
  if (ifarg(2)) { // "GET"
    if (!hoc_is_str_arg(2)) { printf("INTF6 sprob()ERRA: only legit 2nd arg is 'GET'\n"); hxe();
    } else for (j=0;j<ip->dvt;j++) x[j]=(double)ip->sprob[j];
  } else {
    for (j=0;j<ip->dvt;j++) ip->sprob[j]=(unsigned char)x[j];
  }
  }
ENDVERBATIM
}

: turnoff(v1,v2) turn off any connection from a cell in v1 to a cell with number in v2
: a global call that can be called from any INTF6
PROCEDURE turnoff () {
  VERBATIM {
  int nx,ny,i,j,k,dvt; double poid,*x,*y; Point_process **das; unsigned char off;
  ip=IDP; pg=ip->pg;
  nx=vector_arg_px(1,&x);
  ny=vector_arg_px(2,&y);
  if (ifarg(3)) off=(unsigned char)*getarg(3); else off=0;
  for (i=0;i<nx;i++) { 
    lop(pg->ce,(unsigned int)x[i]); 
    dvt=qp->dvt; das=qp->dvi;
    for (j=0;j<dvt;j++) {
      ip=*((id0**) &((das[j]->_prop->dparam)[2])); // sop is *_ppvar[2].pval
      poid=(double)ip->id; // postsyn id
      for (k=0;k<ny;k++) {
        if (poid==y[k]) {
          qp->sprob[j]=off; break;
        }
      }
    }
  }
  }
  ENDVERBATIM
}

VERBATIM 
// gsort2() sorts 2 parallel vectors -- delays and Point_process pointers
int gsort2 (double *db, Point_process **da,int dvt,double *dbs, Point_process **das) {
  int i;
  scr=scrset(dvt);
  for (i=0;i<dvt;i++) scr[i]=i;
  nrn_mlh_gsort(db, scr, dvt, cmpdfn);
  for (i=0;i<dvt;i++) {
    dbs[i]=db[scr[i]];
    das[i]=da[scr[i]];
  }
}
// gsort3() sorts 3 parallel vectors -- delays and Point_process pointers
int gsort3 (double *db, Point_process **da,char* syns,int dvt,double *dbs, Point_process **das,char* synss) {
  int i;
  scr=scrset(dvt);
  for (i=0;i<dvt;i++) scr[i]=i;
  nrn_mlh_gsort(db, scr, dvt, cmpdfn);//sorts indices in scr
  for (i=0;i<dvt;i++) {
    dbs[i]=db[scr[i]];
    das[i]=da[scr[i]];
    synss[i]=syns[scr[i]];
  }
}

// gsort5() sorts 5 parallel vectors -- delays,Point_process pointers,weights,syn types
int gsort5 (double *db, Point_process **da, char* syns, double* w1,double* w2, int dvt,
           double *dbs, Point_process **das,char* synss,double* w1s,double* w2s) {
  int i;
  scr=scrset(dvt);
  for (i=0;i<dvt;i++) scr[i]=i;
  nrn_mlh_gsort(db, scr, dvt, cmpdfn);//sorts indices in scr
  for (i=0;i<dvt;i++) {
    dbs[i]=db[scr[i]];
    das[i]=da[scr[i]];
    synss[i]=syns[scr[i]];
    w1s[i]=w1[scr[i]];
    w2s[i]=w2[scr[i]];
  }
}

static int freedvi2 (struct ID0* jp) {
  if (jp->dvi) {
    free(jp->dvi); free(jp->del); free(jp->sprob); free(jp->syns);
    if(jp->wgain){free(jp->wgain); jp->wgain=0x0;}
    if(jp->peconv){free(jp->peconv); jp->peconv=0x0;}
    if(jp->piconv){free(jp->piconv); jp->piconv=0x0;}
    if(ip->pplasttau){free(ip->pplasttau);ip->pplasttau=0x0;}
    if(ip->pplastinc){free(ip->pplastinc);ip->pplastinc=0x0;}
    if(ip->pplastmaxw){free(ip->pplastmaxw);ip->pplastmaxw=0x0;}
    if(ip->pdope){free(ip->pdope);ip->pdope=0x0;}
    jp->dvt=0; jp->dvi=(Point_process**)0x0; jp->del=(double*)0x0; jp->sprob=(char *)0x0; jp->syns=(char *)0x0;
  }
}
ENDVERBATIM

PROCEDURE freedvi () {
  VERBATIM
  { 
    id0 *jp;
    jp=IDP;
    freedvi2(jp);
  }
  ENDVERBATIM
}

FUNCTION qstats () {
  VERBATIM {
    double stt[3]; int lct,flag; FILE* tfo;
    if (ifarg(1)) {tfo=hoc_obj_file_arg(1); flag=1;} else flag=0;
    lct=cty[IDP->type];
    _lqstats = nrn_event_queue_stats(stt);
    printf("SPIKES: %d (%ld:%ld)\n",IDP->spkcnt,spikes[lct],blockcnt[lct]);
    printf("QUEUE: Inserted %g; removed %g\n",stt[0],stt[2]);
    if (flag) {
      fprintf(tfo,"SPIKES: %d (%ld:%ld);",IDP->spkcnt,spikes[lct],blockcnt[lct]);
      fprintf(tfo,"QUEUE: Inserted %g; removed %g remaining: %g\n",stt[0],stt[2],_lqstats);
    }
  }
  ENDVERBATIM
}

FUNCTION qsz () {
  VERBATIM {
    double stt[3];
    _lqsz = nrn_event_queue_stats(stt);
  }
  ENDVERBATIM
}

PROCEDURE qclr () {
  VERBATIM {
    clear_event_queue();
  }
  ENDVERBATIM
}

: mywmat(from,to,synapse) - return WMAT value from mod side
FUNCTION mywmat () {
  VERBATIM {
  int i,j,k;
  i=(int)*getarg(1);
  if(i<0 || i>=CTYPi){printf("mywmat ERR: arg 1=%d out of bounds (0,%d]\n",i,CTYPi); return -1;}
  j = (int)*getarg(2);
  if(j<0 || j>=CTYPi){printf("mywmat ERR: arg 2=%d out of bounds (0,%d]\n",j,CTYPi); return -1;}
  k = (int)*getarg(3);
  if(k<0 || k>=STYPi){printf("mywmat ERR: arg3=%d out of bounds (0,%d]\n",k,STYPi); return -1;}
  return WMAT(i,j,k);
  }
  ENDVERBATIM  
}

: mywmatpr - print out WMAT from mod side
PROCEDURE mywmatpr () {
  VERBATIM {
  double wm;
  int i,j,k;
  char *ct1,*ct2;
  ip=IDP; pg=ip->pg;
  for(i=0;i<CTYPi;i++) if(ctt(i,&ct1)!=0) {
    for(j=0;j<CTYPi;j++) if(ctt(j,&ct2)!=0) {
      for(k=0;k<STYPi;k++) {
        if((wm=WMAT(i,j,k))>0) {
          printf("wmat[%s][%s][%d]=%g\n",ct1,ct2,k,wm);
        }
      }      
    }
  }
  }
  ENDVERBATIM
}


: intf.cinit() is alternative to jitcondiv() that just sets up cell specific params
PROCEDURE cinit () {
  VERBATIM {
  Symbol *sym; int i,j; unsigned int sz,colid; char *name;

  pg=(postgrp *)calloc(1,sizeof(postgrp));
  colid = (int)*getarg(2);

  if(ppg==0x0) { // initial allocation
    ippgbufsz = 5;
    ppg = (postgrp**) calloc(ippgbufsz,sizeof(postgrp*));
    inumcols = 1;
  } else inumcols++;

  if(inumcols >= ippgbufsz) { // need more memory? then realloc
    ippgbufsz *= 2;
    ppg = realloc((void*)ppg,(size_t)ippgbufsz*sizeof(postgrp*));
  }
  ppg[inumcols-1] = pg;
  pg->col = colid;
  pg->ce =  *hoc_objgetarg(1);

  sym = hoc_lookup("CTYP"); 
  CTYP = (*(hoc_objectdata[sym->u.oboff].pobj));

  if (installed==2.0) { // jitcondiv was previously run
    sz=ivoc_list_count(pg->ce);
    if (sz==pg->cesz && colid==0) printf("\t**** INTF6 WARNING cesz unchanged: INTF6(s) created off-list ****\n");
  } else installed=2.0;
  pg->cesz = ivoc_list_count(pg->ce); if(verbose) printf("cesz=%d\n",pg->cesz);
  pg->lastspk = calloc(pg->cesz,sizeof(double)); // last spike time of each cell
  // not column specific
  CTYPi=HVAL("CTYPi"); STYPi=HVAL("STYPi"); dscrsz=HVAL("scrsz"); dscr=HPTR("scr");
  // column specific
  pg->ix = hoc_pgetarg(3);
  pg->ixe = hoc_pgetarg(4);
  pg->numc = hoc_pgetarg(5); // numc
  if(verbose){printf("CTYPi=%d\n",CTYPi);
    for(i=0;i<CTYPi;i++) printf("ix[%d]=%g, ixe[%d]=%g\n",i,pg->ix[i],i,pg->ixe[i]);}
  if (!pg->ce) {printf("INTF6 cinit() ERRA: ce not found\n"); hxe();}
  if (ivoc_list_count(CTYP)!=CTYPi){
    printf("INTF6 cinit() ERRB: %d %d\n",ivoc_list_count(CTYP),CTYPi); hxe(); }
  for (i=0;i<pg->cesz;i++) { lop(pg->ce,i); qp->pg=pg; } // set all of the pg pointers for now
  printf("Checking for possible seg error in double arrays: CTYPi==%d: ",CTYPi);
  printf("%d %g\n",dscrsz,dscr[dscrsz-1]); // scratch area for doubles
  for (i=0,j=0;i<CTYPi;i++) if (ctt(i,&name)!=0) {
    cty[j]=i; CNAME[j]=name; ctymap[i]=j;
    j++;
    if (j>=CTYPp) {printf("jitcondiv() INTERRA\n"); hxe();}
  }
  CTYN=j; // number of cell types being used
  for (i=0;i<CTYN;i++) printf("%s(%d)=%g ",CNAME[i],cty[i],NUMC(cty[i]));
  printf("\n%d cell types being used in col %d\n",CTYN,colid);
  }
  ENDVERBATIM  
}

: intf.jitcondiv() assigns pointers for hoc symbol storage
PROCEDURE jitcondiv () {
  VERBATIM {
  Symbol *sym; int i,j; unsigned int sz,colid; char *name;

  pg=(postgrp *)calloc(1,sizeof(postgrp));
  colid = (int)*getarg(2);

  if(ppg==0x0) { // initial allocation
    ippgbufsz = 5;
    ppg = (postgrp**) calloc(ippgbufsz,sizeof(postgrp*));
    inumcols = 1;
  } else inumcols++;

  if(inumcols >= ippgbufsz) { // need more memory? then realloc
    ippgbufsz *= 2;
    ppg = realloc((void*)ppg,(size_t)ippgbufsz*sizeof(postgrp*));
  }
  ppg[inumcols-1] = pg;
  pg->col = colid;
  pg->ce =  *hoc_objgetarg(1);

  sym = hoc_lookup("CTYP"); 
  CTYP = (*(hoc_objectdata[sym->u.oboff].pobj));

  if (installed==2.0) { // jitcondiv was previously run
    sz=ivoc_list_count(pg->ce);
    if (sz==pg->cesz && colid==0) printf("\t**** INTF6 WARNING cesz unchanged: INTF6(s) created off-list ****\n");
  } else installed=2.0;
  pg->cesz = ivoc_list_count(pg->ce); if(verbose) printf("cesz=%d\n",pg->cesz);
  pg->lastspk = calloc(pg->cesz,sizeof(double)); // last spike time of each cell

  // not column specific
  CTYPi=HVAL("CTYPi"); STYPi=HVAL("STYPi"); dscrsz=HVAL("scrsz"); dscr=HPTR("scr");

  // column specific
  pg->ix = hoc_pgetarg(3);
  pg->ixe = hoc_pgetarg(4);

  if(verbose){printf("CTYPi=%d\n",CTYPi);
    for(i=0;i<CTYPi;i++) printf("ix[%d]=%g, ixe[%d]=%g\n",i,pg->ix[i],i,pg->ixe[i]);}

  pg->dvg = hoc_pgetarg(5); // div
  pg->numc = hoc_pgetarg(6); // numc
  pg->wmat = hoc_pgetarg(7); // wmat
  pg->wd0 = hoc_pgetarg(8); // wd0
  pg->delm = hoc_pgetarg(9); // delm
  pg->deld = hoc_pgetarg(10); // deld

  if (!pg->ce) {printf("INTF6 jitcondiv ERRA: ce not found\n"); hxe();}
  if (ivoc_list_count(CTYP)!=CTYPi){
    printf("INTF6 jitcondiv ERRB: %d %d\n",ivoc_list_count(CTYP),CTYPi); hxe(); }
  for (i=0;i<pg->cesz;i++) { lop(pg->ce,i); qp->pg=pg; } // set all of the pg pointers for now
  // make sure no seg error:
  printf("Checking for possible seg error in double arrays: CTYPi==%d: ",CTYPi);
  // can access arbitrary member dvg[a][b] using (&dvg[a*CTYPi])[b] or dvg+a*CTYPi+b
  printf("%d %d %d ",DVG(CTYPi-1,CTYPi-1),(int)pg->ix[CTYPi-1],(int)pg->ixe[CTYPi-1]);
  printf("%g %g ",WMAT(CTYPi-1,CTYPi-1,STYPi-1),WD0(CTYPi-1,CTYPi-1,STYPi-1));
  printf("%g %g ",DELM(CTYPi-1,CTYPi-1),DELD(CTYPi-1,CTYPi-1));
  printf("%d %g\n",dscrsz,dscr[dscrsz-1]); // scratch area for doubles
  for (i=0,j=0;i<CTYPi;i++) if (ctt(i,&name)!=0) {
    cty[j]=i; CNAME[j]=name; ctymap[i]=j;
    j++;
    if (j>=CTYPp) {printf("jitcondiv() INTERRA\n"); hxe();}
  }
  CTYN=j; // number of cell types being used
  for (i=0;i<CTYN;i++) printf("%s(%d)=%g ",CNAME[i],cty[i],NUMC(cty[i]));
  printf("\n%d cell types being used in col %d\n",CTYN,colid);
  }
  ENDVERBATIM  
}

: intf.jitrec(vec,tvec)
PROCEDURE jitrec () {
  VERBATIM {
  int i;
  ip=IDP; pg=ip->pg;
  if(verbose>1) printf("jitrec from col %d, ip=%p, pg=%p\n",ip->col,ip,pg);
  if (! ifarg(2)) { // clear with jitrec() or jitrec(0)
    pg->jrmax=0; pg->jridv=0x0; pg->jrtvv=0x0;
    return 0;
  }
  i =   vector_arg_px(1, &pg->jrid); // could just set up the pointers once
  pg->jrmax=vector_arg_px(2, &pg->jrtv);
  pg->jridv=vector_arg(1); pg->jrtvv=vector_arg(2);
  pg->jrmax=vector_buffer_size(pg->jridv);
  if (pg->jrmax!=vector_buffer_size(pg->jrtvv)) {
    printf("jitrec() ERRA: not same size: %d %d\n",i,pg->jrmax); pg->jrmax=0; hxe(); }
  pg->jri=pg->jrj=0; // needs to be set at beginning of run
  }
  ENDVERBATIM
}

: PROCEDURE jitrecreset () {
:   VERBATIM
:   ip=IDP; pg=ip->pg;
:   if(verbose>1) printf("jitrecreset from col %d, ip=%p, pg=%p\n",ip->col,ip,pg);
:   pg->jrj=0; // needs to be set at beginning of run
:   ENDVERBATIM
: }

: intf.scsv()
FUNCTION scsv () {
  VERBATIM {
  int ty=4; int i,j; unsigned int cnt=0;
  ip=IDP; pg=ip->pg;
  name = gargstr(1);
  if ( !(wf1 = fopen(name,"w"))) { printf("Can't open %s\n",name); hxe(); }
  fwrite(&pg->cesz,sizeof(int),1,wf1);
  fwrite(&ty,sizeof(int),1,wf1);
  for (i=0,j=0;i<pg->cesz;i++,j++) { 
    lop(pg->ce,i); 
    if (qp->spkcnt) {
      dscr[j]=(double)(qp->spkcnt); 
      cnt++;
    } else dscr[j]=0.0;
    if (j>=dscrsz) {
      fwrite(dscr,(size_t)sizeof(double),(size_t)dscrsz,wf1);
      fflush(wf1);
      j=0;
    }
  }
  if (j>0) fwrite(dscr,(size_t)sizeof(double),(size_t)j,wf1);
  fclose(wf1);
  _lscsv=(double)cnt;
  }
  ENDVERBATIM
}

: intf.spkcnt(vec[,vec,flag])
: intf.spkcnt(min,max[,vec,flag]) flag=1 means reset all counts to 0
FUNCTION spkcnt () {
  VERBATIM {
  int nx, ny, i,j, ix, c, min, max, flag; unsigned int sum; double *y,*x;
  ip=IDP; pg=ip->pg;
  nx=ny=min=max=flag=0; i=1;
  if (ifarg(i)) {
    if (hoc_is_object_arg(i)) { 
      ny = vector_arg_px(i, &y); i++;
    } else if (ifarg(i+1)) {
      min=(int)*getarg(i); max=(int)*getarg(i+1); i+=2;
    }
  }
  while (ifarg(i)) { // can pick up flag and vector in either order
    if (hoc_is_object_arg(i)) { // output to a vector
      nx = vector_arg_px(i, &x);
    } else flag=(int)*getarg(i);
    i++;
  }
  if (ny) max=ny; else if (max==0) max=pg->cesz; else max+=1; // enter max index wish to graph
  if (nx && nx!=max-min) {
    printf("INTF6 spkcnt() ERR: Vectors not same size %d %d\n",nx,max-min);hxe();}
  for  (i=min, sum=0;i<max;i++) { 
    if (ny) lop(pg->ce,(int)y[i]); else lop(pg->ce,i);
    if (flag==2) sum+=(c=qp->blkcnt); else sum+=(c=qp->spkcnt);
    if (nx) x[i]=(double)c;
    if (flag==1) qp->spkcnt=qp->blkcnt=0;
  }
  _lspkcnt=(double)sum;
  }
  ENDVERBATIM
}

:** probejcd()
PROCEDURE probejcd () {
  VERBATIM {  int i,a[4];
    ip=IDP; pg=ip->pg;
    for (i=1;i<=3;i++) a[i]=(int)*getarg(i);
    printf("CTYPi: %d, STYPi: %d, ",CTYPi,STYPi);
    // printf("div: %d, ix: %d, ixe: %d, ",DVG(a[1],a[2]),(int)ix[a[1]],(int)ixe[a[1]]);
    printf("wmat: %g, wd0: %g\n",WMAT(a[1],a[2],a[3]),WD0(a[1],a[2],a[3]));
  }
  ENDVERBATIM  
}

:** randspk() sets next to next val in vector, this vector is handled globally
PROCEDURE randspk () {
  VERBATIM 
  ip=IDP; pg=ip->pg;
  if (ip->rvi > ip->rve) { // pointers go from rvi to rve inclusive
    ip->input=0;           // turn off
    //nxt=-1.; // MR: Commented this out because it kills off the net_send / NET_RECEIVE feedback
                // loop once the input spike vecs have finished. If we then push more spikes to the net
                // at a later time, they'll never get picked up if nxt=-1 as the loop is dead.
  } else if (t==0) {     // initialization
    nxt=pg->vsp[ip->rvi];
    EXSY=pg->sysp[ip->rvi]; // synapse target for external input
    WEX=pg->wsp[ip->rvi++]; // weight of external input
  } else {     // absolute times in vector -> interval
    while ((nxt=pg->vsp[ip->rvi++]-t)<=1e-6) { 
      if (ip->rvi-1 > ip->rve) { printf("randspk() ERRA: "); chk(2.); hxe(); }
    }
    EXSY=pg->sysp[ip->rvi-1]; // rvi was incremented
    WEX=pg->wsp[ip->rvi-1]; // rvi was incremented    
  }
  ENDVERBATIM
  : net_send(nxt,2) : can only be called from INITIAL or NET_RECEIVE blocks
}

:** vers gives version
PROCEDURE vers () {
  printf("$Id: intf6.mod,v 1.100 2012/04/05 22:38:25 samn Exp $\n")
}

:** val(t,tstart) fills global vii[] to pass values back to record() (called from record())
VERBATIM
double val (double xx, double ta) { 
  vii[1]=VAM*EXP(-(xx - ta)/tauAM);
  vii[2]=VNM*EXP(-(xx - ta)/tauNM);
  vii[3]=VGA*EXP(-(xx - ta)/tauGA);

  vii[5]=AHP*EXP(-(xx - ta)/tauahp);
  vii[8]=VAM2*EXP(-(xx -ta)/tauAM2);
  vii[9]=VNM2*EXP(-(xx - ta)/tauNM2);
  vii[10]=VGA2*EXP(-(xx - ta)/tauGA2);
  vii[6]=vii[1]+vii[2]+vii[3]+vii[4]+vii[5]+vii[8]+vii[9]+vii[10];
  vii[7]=VTH + (VTHR-VTH)*EXP(-(xx-trrs)/tauRR) - RMP; // sub RMP, since gets added later
}
ENDVERBATIM

:** valps(t,tstart) like val but builds voltages for pop spike
VERBATIM
double valps (double xx, double ta) { 
  vii[1]=VAM*EXP(-(xx - ta)/tauAM);
  vii[2]=VNM*EXP(-(xx - ta)/tauNM);
  vii[3]=VGA*EXP(-(xx - ta)/tauGA);

  vii[8]=VAM2*EXP(-(xx - ta)/tauAM2);
  vii[9]=VNM2*EXP(-(xx - ta)/tauNM2);
  vii[10]=VGA2*EXP(-(xx - ta)/tauGA2);
  vii[6]=vii[1]+vii[2]-vii[3]+vii[8]+vii[9]-vii[10];
}
ENDVERBATIM

:** record() stores values since last tg into appropriate vecs
PROCEDURE record () {
  VERBATIM {
  int i,j,k,nz; double ti;
  vp = SOP;
  if(!vp) {printf("**** record ERRA: vp=NULL!\n"); return 0;}
  if (tg>=t) return 0;
  if (ip->record==1) {
    while ((int)vp->p >= (int)vp->size-(int)((t-tg)/vdt)-10) { 
      vp->size*=2;
      for (k=0;k<NSV;k++) if (vp->vv[k]!=0x0) vp->vvo[k]=vector_newsize(vp->vv[k], vp->size);
      // printf("**** WARNING expanding recording room to %d (type%d id%d at %g)****\n",vp->size,IDP->type,IDP->id,t);
    }
  } else if ((int)vp->p > (int)vp->size-(int)((t-tg)/vdt)) { // shift if record==2
    nz=(int)((t-tg)/vdt);
    for (k=0;k<NSV;k++) if (vp->vv[k]!=0x0) {
      if (nz>vp->size) {pid(); printf("Record WARNING: vec too short: %d %d\n",nz,vp->size);
        vp->p=0;
      } else {
        for (i=nz,j=0; i<vp->size; i++,j++) vp->vvo[k][j]=vp->vvo[k][i];
        vp->p=vp->size-nz;
      }
    }
  }
  for (ti=tg;ti<=t && vp->p < vp->size;ti+=vdt,vp->p++) { 
    val(ti,tg);  
    if (vp->vvo[0]!=0x0) vp->vvo[0][vp->p]=ti;
    for (k=1;k<NSV-1;k++) if (vp->vvo[k]!=0x0) { // not nil pointer
      vp->vvo[k][vp->p]=vii[k]+RMP;
    }
    for (;k<NSV;k++) if (vp->vvo[k]!=0x0) { // not nil pointer
      vp->vvo[k][vp->p]=vii[k]; 
    }
  }
  tg=t;
  }
  ENDVERBATIM
}

:** recspk() records a spike by writing a 10 into the main VM vector
PROCEDURE recspk (x) {
  VERBATIM { 
  vp = SOP;
  record();
  if (vp->p > vp->size || vp->vvo[6]==0) return 0; 
  if (vp->p > 0) {
    if (vp->vvo[0]!=0x0) vp->vvo[0][vp->p-1]=_lx;
    vp->vvo[6][vp->p-1]=spkht; // the spike
  } else {
    if (vp->vvo[0]!=0x0) vp->vvo[0][0]=_lx; 
    vp->vvo[6][0]=spkht; // the spike
  }
  tg=_lx;
  }
  ENDVERBATIM
}

:** recclr() clear the vectors pointers
PROCEDURE recclr () {
  VERBATIM 
  {int k;
  if (IDP->record) {
    if (SOP!=nil) {
      vp = SOP;
      vp->size=0; vp->p=0;
      for (k=0;k<NSV;k++) { vp->vv[k]=nil; vp->vvo[k]=nil; }
    } else printf("INTF6 recclr ERR: nil pointer\n");
  }
  IDP->record=0;
  }
  ENDVERBATIM 
}

:** recfree() free the vpt pointer memory
PROCEDURE recfree () {
  VERBATIM
  if (SOP!=nil) {
    free(SOP);
    SOP=nil;
  } else printf("INTF6 recfree ERR: nil pointer\n");
  IDP->record=0;
  ENDVERBATIM
}

:** initvspks() sets up vector from which to read random spike times 
: this is a global procedure to set up pieces of a global vector
: all cells share one vector but read from different locations
: (CHANGED from intervals and global proc in v224)
: intf.initvspks(indices, times , weights, synapse types)
PROCEDURE initvspks () {
  VERBATIM
  {int max, i,err;
    double last,lstt;
    ip=IDP; pg=ip->pg;
    if (! ifarg(1)) {printf("Return initvspks(indices,times,weights,syntypes)\n"); return 0.;}
    if(verbose>1) printf("initvspks: col=%d, ip=%p, pg=%p, pg->isp=%p\n",ip->col,ip,pg,pg->isp);
    if (pg->isp!=NULL) clrvspks();
    ip=IDP; pg=ip->pg; err=0;
    i = vector_arg_px(1, &pg->isp); // could just set up the pointers once
    max=vector_arg_px(2, &pg->vsp);
    if (max!=i) {err=1; printf("initvspks ERR: vecs of different size\n");}
    if (max==0) {err=1; printf("initvspks ERR: vec not initialized\n");}
    max=vector_arg_px(3, &pg->wsp);
    if (max!=i) {err=1; printf("initvspks ERR: 3rd vec is of different size\n");}
    max=vector_arg_px(4, &pg->sysp);
    if (max!=i) {err=1; printf("initvspks ERR: 4th vec is of different size\n");}
    pg->vspn=max;
    if (!pg->ce) {printf("Need global ce for initvspks() since intf.mod501\n"); hxe();}
    for (i=0,last=-1; i<max; ) { // move forward to first
      if (pg->isp[i]!=last) { // new one
        lop(pg->ce,(unsigned int)pg->isp[i]);
        qp->rvb=qp->rvi=i;
        qp->vinflg=qp->input=1;
        last=pg->isp[i];
        lstt=pg->vsp[i];
        i++;
      }
      for (; i<max && pg->isp[i] == last; i++) { // move forward to last
        if (pg->vsp[i]<=lstt) { pg->vsp[i]=lstt+0.00001; // CK: was err=1; this avoid monotonic error
          printf("initvspks ERR: nonmonotonic for cell#%d: %g %g\n",qp->id,lstt,pg->vsp[i]); }
          lstt=pg->vsp[i];
      }
      qp->rve=i-1;
      if (subsvint>0) { 
        pg->vsp[qp->rve] = pg->vsp[qp->rvb]+subsvint;
        pg->wsp[qp->rve] = pg->wsp[qp->rvb];
      }
      if (err) { qp->rve=0; hxe(); }
    }
  }
  ENDVERBATIM
}

:** shock() reads random spike times from same db as initvspks() but just sends a single shock
: to each listed cell
: this is a global procedure that calls multiple cells
PROCEDURE shock () {
  VERBATIM 
  {int max, i,err;
    double last, lstt, *isp, *vsp, *wsp;
    printf("WARNING: This routine appears to be defunct -- please check code in intf6.mod\n");
    if (! ifarg(1)) {printf("Return shock(ivspks,vspks,wvspks)\n"); return 0.;}
    ip=IDP; pg=ip->pg; err=0;
    i = vector_arg_px(1, &isp); // could just set up the pointers once
    max=vector_arg_px(2, &vsp);
    if (max!=i) {err=1; printf("shock ERR: vecs of different size\n");}
    if (max==0) {err=1; printf("shock ERR: vec not initialized\n");}
    max=vector_arg_px(3, &wsp);
    if (max!=i) {err=1; printf("shock ERR: 3rd vec is of different size\n");}
    pg->vspn=max;
    if (!pg->ce) {printf("Need global ce for shock()\n"); hxe();}
    for (i=0,last=-1; i<max; ) { // move forward to first
      if (isp[i]!=last) { // skip any redund indices
        lop(pg->ce,(unsigned int)isp[i]);
        WEX=-1e9; // code for shock
        EXSY=AM;  // set to AMPA, though doesn't matter for single shock
  #if defined(t)
        net_send((void**)0x0, wts,pmt,t+vsp[i],2.0); // 2 is randspk flag
  #else
        net_send((void**)0x0, wts,pmt,vsp[i],2.0); // 2 is randspk flag
  #endif
        i++;
      }
    }
  }
  ENDVERBATIM
}

PROCEDURE clrvspks () {
 VERBATIM {
 unsigned int i;
 ip=IDP; pg=ip->pg;
 if(verbose>1) printf("clrvspks: col=%d, ip=%p, pg=%p, pg->isp=%p\n",ip->col,ip,pg,pg->isp);
 for (i=0; i<pg->cesz; i++) {
   lop(pg->ce,i);
   qp->vinflg=0;
 }   
 }
 ENDVERBATIM
}

: trvsp gets called globally to go through the vector
: first pass (arg 1) it replaces terminal values with 1e9
: second pass (arg 2) it replaces terminal values with first+subsvint
PROCEDURE trvsp ()
{
  VERBATIM 
  int i, flag; 
  double ind, t0;
  ip=IDP; pg=ip->pg;
  flag=(int) *getarg(1);
  if (subsvint==0.) {printf("trvsp"); return(0.);}
  ind=pg->isp[0]; t0=pg->vsp[0];
  if (flag==1) {
    for (i=0; i<pg->vspn; i++) {
      if (pg->isp[i]!=ind) {
        pg->vsp[i-1]=1.e9;
        ind=pg->isp[i];
      }
    }
    pg->vsp[pg->vspn-1]=1.e9;
  } else if (flag==2) {
    for (i=0; i<pg->vspn; i++) {
      if (pg->isp[i]!=ind) {
        pg->vsp[i-1]=t0+subsvint;
        ind=pg->isp[i]; t0=pg->vsp[i];
      }
    }
    pg->vsp[pg->vspn-1]=t0+subsvint;
  } else {printf("trvsp flag %d not recognized\n",flag); hxe();}
  ENDVERBATIM
}

:** initjttr() sets up vector from which to read jitter 
: -- key jtt to avoid confusion with jitcon=='just in time connection'
: this is a global not a range procedure -- just call once
PROCEDURE initjttr () {
  VERBATIM 
  {int max, i, err=0;
    ip=IDP; pg=ip->pg;
    pg->jtpt=0;
    if (! ifarg(1)) {printf("Return initjttr(vec)\n"); return(0.);}
    max=vector_arg_px(1, &jsp);
    if (max==0) {err=1; printf("initjttr ERR: vec not initialized\n");}
    for (i=0; i<max; i++) if (jsp[i]<=0) {err=1;
      printf("initjttr ERR: vec should be >0: %g\n",jsp[i]);}
    if (err) { jsp=nil; pg->jtmax=0.; return(0.); }// hoc_execerror("",0);
    if (max != pg->jtmax) {
      printf("WARNING: resetting jtmax_INTF6 to %d\n",max); pg->jtmax=max; }
  }
  ENDVERBATIM
}

:* internal routines
VERBATIM

//** getlp(LIST,ITEM#) sets qp: take object from ob list @ index i and return pointer
// modeled on vector_arg_px(): picks up obj from list and resolves pointers
static id0* getlp (Object *ob, unsigned int i) {
  Object *lb; id0* myp;
  lb = ivoc_list_item(ob, i);
  if (! lb) { printf("INTF6:getlp %d exceeds %d for list ce\n",i,pg->cesz); hxe();}
  pmt=ob2pntproc(lb);
  myp=*((id0**) &((pmt->_prop->dparam)[2])); // #define sop *_ppvar[2].pval
  return myp;
}

//** lop(LIST,ITEM#) sets qp: take object from ob list @ index i and assign pointer to GLOBAL qp pointer
// modeled on vector_arg_px(): picks up obj from list and resolves pointers
static id0* lop (Object *ob, unsigned int i) {
  Object *lb;
  lb = ivoc_list_item(ob, i);
  if (! lb) { printf("INTF6:lop %d exceeds %d for list ce\n",i,pg->cesz); hxe();}
  pmt=ob2pntproc(lb);
  qp=*((id0**) &((pmt->_prop->dparam)[2])); // #define sop *_ppvar[2].pval
  return qp;
}

//*** lopr(LIST,ITEM#) sets qp and RANGE vars - same as lop() but also does RANGE
static id0* lopr (Object *ob, unsigned int i) {
  id0* myp;
  myp = lop(ob,i);
  _hoc_setdata(pmt); // pmt is another global
  return myp;
}

// use stoppo() as a convenient conditional breakpoint in gdb (gdb watching is too slow)
int stoppo () {
}

//** ctt(ITEM#) find cells that exist by name
static int ctt (unsigned int i, char** name) {
  Object *lb;
  if (NUMC(i)==0) return 0; // none of this cell type
  lb = ivoc_list_item(CTYP, i);
  if (! lb) { printf("INTF6:ctt %d exceeds %d for list CTYP\n",i,CTYPi); hxe();}
  {*name=*(lb->u.dataspace->ppstr);}
  return (int)NUMC(i);
}
ENDVERBATIM


PROCEDURE test () {
  VERBATIM
  char *str; int x;
  x=ctt(7,&str); 
  printf("%s (%d)\n",str,x);
  ENDVERBATIM
}

: lof can find object information
PROCEDURE lof () {
VERBATIM {
  Object *ob; int num,i,ii,j,k,si,nx;  double *vvo[7], *par; void *vv[7];
  ob = *(hoc_objgetarg(1));
  si=(int)*getarg(2);
  num = ivoc_list_count(ob);
  if (num!=7) { printf("INTF6 lof ERR %d>7\n",num); hxe(); }
  for (i=0;i<num;i++) { 
    j = list_vector_px3(ob, i, &vvo[i], &vv[i]);
    if (i==0) nx=j;
    if (j!=nx) { printf("INTF6 lof ERR %d %d\n",j,nx); hxe(); }
  }
  //  for (i=ix[si],ii=0;i<=ixe[si] && ii<nx;i++,ii++) {
  //   vvo[0][ii]=(double)i;
  //   par=lop(ce,i);
  //   for (j=20,k=1;j<25;j++,k++) { // NB these could move: Vm,VAM,VNM,VGA
  //     vvo[k][ii]=par[j];
  //   }
  // }
 }
ENDVERBATIM
}

:* initinvl() sets up vector from which to read intervals
: this is a global not a range procedure -- just call once
PROCEDURE initinvl () {
  printf("initinvl() NOT BEING USED\n")
}

: invlflag() used internally; can't set from here; use initinvl() and range invlset()
FUNCTION invlflag () {
  VERBATIM
  ip=IDP; pg=ip->pg;
  if (ip->invl0==1 && invlp==nil) { // err
    printf("INTF6 invlflag ERR: pointer not initialized\n"); hoc_execerror("",0); 
  }
  _linvlflag= (double)ip->invl0;
  ENDVERBATIM
}

:** shift() returns the appropriate shift
FUNCTION shift (vl) { 
  VERBATIM   
  double expand, tmp, min, max;
//if (invlp==nil) {printf("INTF6 invlflag ERRa: pointer not initialized\n"); hoc_execerror("",0);}
  if ((t<(invlt-invl)+invl/2) && invlt != -1) { // don't shift if less than halfway through
    _lshift=0.;  // flag for no shift
  } else {
    expand = -(_lvl-(-65))/20; // expand positive if hyperpolarized
    if (expand>1.) expand=1.; if (expand<-1.) expand=-1.;
    if (expand>0.) { // expand interval
      max=1.5*invl;
      tmp=oinvl+0.8*expand*(max-oinvl); // the amount we can add to the invl
    } else {
      min=0.5*invl; 
      tmp=oinvl+0.8*expand*(oinvl-min); // the amount we can reduce current invl
    }
    if (invlt+tmp<t+2) { // getting too near spike time
      _lshift=0.;
    } else {
      oinvl=tmp; // new interval
      _lshift=invlt+oinvl;
    }
  }
  ENDVERBATIM
}

:* recini() called from INITIAL block to set vp->p to zero and open up vectors
PROCEDURE recini () {
  VERBATIM 
  { int k;
  if (SOP==nil) { 
    printf("INTF6 record ERR: pointer not initialized\n"); hoc_execerror("",0); 
  } else {
    vp = SOP;
    vp->p=0;
    // open up the vector maximally before writing into it; will correct size in fini
    for (k=0;k<NSV;k++) if (vp->vvo[k]!=0) vector_resize(vp->vv[k], vp->size);
  }}
  ENDVERBATIM
}

:** fini() to finish up recording -- should be called from FinishMisc()
PROCEDURE fini () {
  VERBATIM 
  {int k;
  // initialization for next round, this will not be set if job terminates prematurely
  IDP->rvi=IDP->rvb;  // -- see vinset()
  if (IDP->wrec) { wrecord(1e9); }
  if (IDP->record) {
    record(); // finish up
    for (k=0;k<NSV;k++) if (vp->vvo[k]!=0) { // not nil pointer
      vector_resize(vp->vv[k], vp->p);
    }
  }}
  ENDVERBATIM
}

:** chk([flag]) with flag=1 prints out info on the record structure
:                    flag=2 prints out info on the global vectors
PROCEDURE chk (f) {
  VERBATIM 
  {int i,lfg;
  lfg=(int)_lf;
  ip=IDP; pg=ip->pg;
  printf("ID:%d; typ: %d; rec:%d wrec:%d inp:%d jtt:%d invl:%d\n",ip->id,ip->type,ip->record,ip->wrec,ip->input,ip->jttr,ip->invl0);
  if (lfg==1) {
    if (SOP!=nil) {
      vp = SOP;
      printf("p %d size %d tg %g\n",vp->p,vp->size,tg);
      for (i=0;i<NSV;i++) if (vp->vv[i]) printf("%d %x %x;",i,(unsigned int)vp->vv[i],(unsigned int)vp->vvo[i]);
    } else printf("Recording pointers not initialized");
  }
  if (lfg==2) { 
    printf("Global vectors for input and jitter (jttr): \n");
    if (pg->vsp!=nil) printf("VSP: %x (%d/%d-%d)\n",(unsigned int)pg->vsp,ip->rvi,ip->rvb,ip->rve); else printf("no VSP\n");
    if (jsp!=nil) printf("JSP: %x (%d/%d)\n",(unsigned int)jsp,pg->jtpt,pg->jtmax); else printf("no JSP\n");
  }
  if (lfg==3) { 
    if (pg->vsp!=nil) { printf("VSP: (%d/%d-%d)\n",ip->rvi,ip->rvb,ip->rve); 
      for (i=ip->rvb;i<=ip->rve;i++) printf("%d:%g  ",i,pg->vsp[i]);
      printf("\n");
    } else printf("no VSP\n");
  }
  if (lfg==4) {  // was used to give invlp[],invlmax
  }
  if (lfg==5) { 
    printf("wwpt %d wwsz %d\n WW vecs: ",wwpt,wwsz);
    printf("wwwid %g wwht %d nsw %g\n WW vecs: ",wwwid,(int)wwht,nsw);
    for (i=0;i<NSW;i++) printf("%d %x %x;",i,(unsigned int)ww[i],(unsigned int)wwo[i]);
  }}
  ENDVERBATIM
}

:** id() and pid() identify the cell -- printf and function return
FUNCTION pid () {
  VERBATIM 
  printf("INTF6%d(%d/%d@%g) ",IDP->id,IDP->type,IDP->col,t);
  _lpid = (double)IDP->id;
  ENDVERBATIM
}

: intra-column identifier for cell
FUNCTION id () {
  VERBATIM
  if (ifarg(1)) IDP->id = (unsigned int) *getarg(1);
  _lid = (double)IDP->id;
  ENDVERBATIM
}

FUNCTION type () {
  VERBATIM
  if (ifarg(1)) IDP->type = (unsigned char) *getarg(1);
  _ltype = (double)IDP->type;
  ENDVERBATIM
}

: column identifier for cell
FUNCTION col () {
  VERBATIM 
  ip = IDP; 
  if (ifarg(1)) ip->col = (unsigned int) *getarg(1);
  _lcol = (double)ip->col;
  ENDVERBATIM
}

: global identifier for cell
FUNCTION gid () {
  VERBATIM 
  ip = IDP; 
  if (ifarg(1)) ip->gid = (unsigned int) *getarg(1);
  _lgid = (double)ip->gid;
  ENDVERBATIM
}

FUNCTION dbx () {
  VERBATIM 
  ip = IDP; 
  if (ifarg(1)) ip->dbx = (unsigned char) *getarg(1);
  _ldbx = (double)ip->dbx;
  ENDVERBATIM
}

:** initrec(name,vec) sets up recording of name (see varnum for list) into a vector
PROCEDURE initrec () {
  VERBATIM 
  {int i;
  name = gargstr(1);
  if (SOP==nil) { 
    IDP->record=1;
    SOP = (vpt*)ecalloc(1, sizeof(vpt));
    SOP->size=0;
  }
  if (IDP->record==0) {
    recini();
    IDP->record=1;
  }
  vp = SOP;
  i=(int)varnum();
  if (i==-1) {printf("INTF6 record ERR %s not recognized\n",name); hoc_execerror("",0); }
  vp->vv[i]=vector_arg(2);
  vector_arg_px(2, &(vp->vvo[i]));
  if (vp->size==0) { vp->size=(unsigned int)vector_buffer_size(vp->vv[i]);
  } else if (vp->size != (unsigned int)vector_buffer_size(vp->vv[i])) {
    printf("INTF6 initrec ERR vectors not all same size: %d vs %d",vp->size,vector_buffer_size(vp->vv[i]));
    hoc_execerror("", 0); 
  }} 
  ENDVERBATIM
}

:** varnum(statevar_name) returns index number associated with particular variable name
: called by initrec() using global name
FUNCTION varnum () { LOCAL i
  i=-1
  VERBATIM
  if (strcmp(name,"time")==0)      { _li=0.;
  } else if (strcmp(name,"VAM")==0) { _li=1.;
  } else if (strcmp(name,"VNM")==0) { _li=2.;
  } else if (strcmp(name,"VGA")==0) { _li=3.;
  } else if (strcmp(name,"AHP")==0) { _li=5.;
  } else if (strcmp(name,"V")==0) { _li=6.;
  } else if (strcmp(name,"VM")==0) { _li=6.; // 2 names for V
  } else if (strcmp(name,"VTHC")==0) { _li=7.;
  } else if (strcmp(name,"VAM2")==0) { _li=8.;
  } else if (strcmp(name,"VNM2")==0) { _li=9.;
  } else if (strcmp(name,"VGA2")==0) { _li=10.;
  }
  ENDVERBATIM
  varnum=i
}

:** vecname(INDEX) prints name when given an index
PROCEDURE vecname () {
  VERBATIM
  int i; 
  i = (int)*getarg(1);
  if (i==0)      printf("time\n");
  else if (i==1) printf("VAM\n");
  else if (i==2) printf("VNM\n");
  else if (i==3) printf("VGA\n");
  else if (i==5) printf("AHP\n");
  else if (i==6) printf("V\n");
  else if (i==7) printf("VTHC\n");
  else if (i==8) printf("VAM2\n");
  else if (i==9) printf("VNM2\n");
  else if (i==10) printf("VGA2\n");
  ENDVERBATIM
}

:** initwrec(name,vec) sets up recording of sim field potential
PROCEDURE initwrec () {
  VERBATIM 
  {int i, k, num, cap;  Object* ob;
    ob =   *hoc_objgetarg(1); // list of vectors
    num = ivoc_list_count(ob);
    if (num>NSW) { printf("INTF6 initwrec() WARN: can only store %d ww vecs\n",NSW); hxe();}
    nsw=(double)num;
    for (k=0;k<num;k++) {
      cap = list_vector_px2(ob, k, &wwo[k], &ww[k]);
      if (k==0) wwsz=cap; else if (wwsz!=cap) {
        printf("INTF6 initwrec ERR w-vecs size err: %d,%d,%d",k,wwsz,cap); hxe(); }
    }
  }
  ENDVERBATIM
}

: popspk() is paste on gaussian for a pop spk: with vdt=0.1 -20 to 20 is 4 ms
: needs to be above location where is actively accessed
PROCEDURE popspk (x) {
  TABLE Psk DEPEND wwwid,wwht FROM -40 TO 40 WITH 81
  Psk = -wwht*exp(-2.*x*x/wwwid/wwwid)
}

PROCEDURE pskshowtable () {
  VERBATIM 
  int j;
  printf("_tmin_popspk:%g -_tmin_popspk:%g\n",_tmin_popspk,-_tmin_popspk);
  for (j=0;j<=-2*(int)_tmin_popspk+1;j++) printf("%g ",_t_Psk[j]);
  printf("\n");
  ENDVERBATIM 
}

:** wrecord() records voltages onto single global vector
PROCEDURE wrecord (te) {
  VERBATIM 
  {int i,j,k,max,wrp; double ti,scale;
  for (i=0;i<WRNUM && (wrp=(int)IDP->wreci[i])>-1;i++) {
    // wrp: index for multiple field recordings
    scale=(double)IDP->wscale[i];
    if (_lte<1.e9) { // a spike recording
      if (scale>0) {
        max=(int)_tmin_popspk; // max of table max=-min
        k=-(int)floor((_lte-rebeg)/vdt+0.5);
        for (j= -max;j<=max && k+j>0 && k+j<wwsz;j++) {
          wwo[wrp][k+j] += scale*_t_Psk[j+max]; // direct copy from the Psk table
        }
      }
    } else if (twg>=t) { return 0;
    } else {
      for (ti=twg,k=(int)floor((twg-rebeg)/vdt+0.5);ti<=t && k<wwsz;ti+=vdt,k++) { 
        valps(ti,twg);  // valps() for pop spike calculation
        wwo[wrp][k]+=vii[6]*lfpscale;
        if (IDP->dbx==-1) printf("%g:%g ",vii[6],wwo[wrp][k]);
      }
    }
  }
  if (_lte==1.e9) twg=ti;
  }
  ENDVERBATIM
}

: backward compatibility -- note that index was 1-offset; convert to 0 offset here
: wrec() -- return value in wrec0
: wrec(VAL) -- set wrec0
: wrec(VAL,SCALE) -- set wrecIND and scaling for wrecIND
FUNCTION wrec () {
  VERBATIM
  { int k,ix;
  ip=IDP; 
  if (ifarg(1)) {
    ix=(int)*getarg(1);
    if (ix>=1) {
      if (ix-1>=nsw) {
        printf("Attempt to save into ww[%d] but only have %d\n",ix-1,(int)nsw); hxe();}
      ip->wrec=1;
      ip->wreci[0]=(char)ix-1;
      ip->wscale[0]=1.; // default
      if (ifarg(2)) ip->wscale[0]= (float)*getarg(2); 
    } else if (ix<=0) {
      ip->wrec=0;
      for (k=0;k<WRNUM;k++) { ip->wreci[k]=-1; ip->wscale[k]=-1.0; }
    } else {printf("INTF6 wrec ERR flag(0/1) %d\n",ip->wrec); hxe();
    }
  }
  _lwrec=(double)ip->wrec;
  }
  ENDVERBATIM
}

: wrc() -- return value in wrec0
: wrc(VAL) -- set wrec0
: wrc(IND,SCALE) -- set wrec0 and scaling for wrec0
FUNCTION wrc () {
  VERBATIM
  { int i,ix;
  ip=IDP; 
  if (ifarg(1)) {  // 1 or 2 args
    ix=(int)*getarg(1);
    if (ix<0) {
      ip->wrec=0;
      for (i=0;i<WRNUM;i++) { ip->wreci[i]=-1; ip->wscale[i]=-1.0; }
    } else {
      for (i=0;i<WRNUM && ip->wreci[i]!=-1 && ip->wreci[i]!=ix;i++) {};
      if (i==WRNUM) {
        pid(); printf("INFT wrc() ERR: out of wreci pointers (max %d)\n",WRNUM); hxe();}
      if (ix>=nsw) {printf("Attempt to save into ww[%d] but only have %d\n",ix,(int)nsw); hxe();}
      ip->wrec=1; 
      ip->wreci[i]=ix;
      if (ifarg(2)) ip->wscale[i]=(float)*getarg(2); else ip->wscale[i]=1.0;
    }
  } else {
    for (i=0;i<WRNUM;i++) printf("%d:%g ",ip->wreci[i],ip->wscale[i]);
    printf("\n");
  }
  _lwrc=(double)ip->wrec;
  }
  ENDVERBATIM
}

FUNCTION wwszset () {
  VERBATIM
  if (ifarg(1)) wwsz = (unsigned int) *getarg(1);
  _lwwszset=(double)wwsz;
  ENDVERBATIM
}

:** wwfree()
FUNCTION wwfree () {
  VERBATIM
  int k;
  IDP->wrec=0;
  wwsz=0; wwpt=0; nsw=0.;
  for (k=0;k<NSW;k++) { ww[k]=nil; wwo[k]=nil; }
  ENDVERBATIM
}

:** jttr() reads out of a noise vector (call from NET_RECEIVE block)
FUNCTION jttr () {
  VERBATIM 
  ip=IDP; pg=ip->pg;
  if (pg->jtmax>0 && pg->jtpt>=pg->jtmax) {  
    pg->jtpt=0;
    printf("Warning, cycling through jttr vector at t=%g\n",t);
  }
  if (pg->jtmax>0) _ljttr = jsp[pg->jtpt++]; else _ljttr=0;
  ENDVERBATIM
}

:** global_init() initialize globals shared by all INTF6s
PROCEDURE global_init () {
  popspk(0) : recreate table if any change in wid or ht
  VERBATIM 
  { int i,j,k,c; double stt[3];
  if (nsw>0. && wwo[0]!=0) { // do just once
    printf("Initializing ww to record for %g (%g)\n",vdt*wwsz,vdt);
    wwpt=0;
    for (k=0;k<(int)nsw;k++) {
      vector_resize(ww[k], wwsz);
      for (j=0;j<wwsz;j++) wwo[k][j]=0.;
    }
  }
  errflag=0;
  for (i=0;i<CTYN;i++) blockcnt[cty[i]]=spikes[cty[i]]=0;
  for(c=0;c<inumcols;c++) {
    pg=ppg[c]; if(!pg) continue;
    if (pg->jridv) { pg->jri=pg->jrj=0; vector_resize(pg->jridv, pg->jrmax); vector_resize(pg->jrtvv, pg->jrmax); }
    pg->spktot=0;
    pg->jtpt=0;
    pg->eventtot=0;
  }
  }
  ENDVERBATIM
}

PROCEDURE global_fini () {
  VERBATIM
  int c,k;
  for (k=0;k<(int)nsw;k++) vector_resize(ww[k], (int)floor(t/vdt+0.5));
  for(c=0;c<inumcols;c++) {
    pg=ppg[c]; if(!pg) continue;
    if (pg->jridv && pg->jrj<pg->jrmax) {
      vector_resize(pg->jridv, pg->jrj); 
      vector_resize(pg->jrtvv, pg->jrj);
    }
  }
  ENDVERBATIM
}

:* setting and getting flags: fflag, record,input,jttr
FUNCTION fflag () { fflag=1 }
FUNCTION thrh () { thrh=VTH-RMP }
: reflag() used internally; can't set from here; use recinit()
FUNCTION recflag () { 
  VERBATIM
  _lrecflag= (double)IDP->record;
  ENDVERBATIM
}

: vinflag() used internally; can't set from here; use global initvspks() and range vinset()
FUNCTION vinflag () {
  VERBATIM
  ip=IDP; pg=ip->pg;
  if (ip->vinflg==0 && pg->vsp==nil) { // do nothing
  } else if (ip->vinflg==1 && ip->rve==-1) {
    printf("INTF6 vinflag ERR: pointer not initialized\n"); hoc_execerror("",0); 
  } else if (ip->rve >= 0) { 
    if (pg->vsp==nil) {
      printf("INTF6 vinflag ERR1: pointer not initialized\n"); hoc_execerror("",0); 
    }
    ip->rvi=ip->rvb;
    ip->input=1;
  }
  _lvinflag= (double)ip->vinflg;
  ENDVERBATIM
}

:** flag(name,[val,setall]) set or get a flag
:   flag(name,vec) fill vec with flag value from all the cells
: seek names from iflags[] and look at location &ip->type -- beginning of flags
FUNCTION flag () {
  VERBATIM
  char *sf; static int ix,fi,setfl,nx; static unsigned char val; static double *x, delt;
  ip=IDP; pg=ip->pg;
  if (FLAG==OK) { // callback -- DO NOT SET FROM HOC
    FLAG=0.;
    if (stoprun) {slowset=0; return 0.0;}
    if (IDP->dbx==-1)printf("slowset fi:%d ix:%d ss:%g delt:%g t:%g\n",fi,ix,slowset,delt,t);
    if (t>slowset || ix>=pg->cesz) {  // done
      printf("Slow-setting of flag %d finished at %g: (%d,%g,%g)\n",fi,t,ix,delt,slowset); 
      slowset=0.; return 0.0;
    }
    if (ix<pg->cesz) {
      lop(pg->ce,ix);
      (&qp->type)[fi]=((fi>=iflneg)?(char)x[ix]:(unsigned char)x[ix]);
      ix++;
      #if defined(t)
      net_send((void**)0x0, wts,tpnt,t+delt,OK); // OK is flag() flag
      #else
      net_send((void**)0x0, wts,tpnt,delt,OK);
      #endif
    }
    return 0.0;
  }  
  if (slowset>0 && ifarg(3)) {
    printf("INTF6 flag() slowset ERR; attempted set during slowset: fi:%d ix:%d ss:%g delt:%g t:%g",\
           fi,ix,slowset,delt,t); 
    return 0.0;
  }
  ip = IDP; setfl=ifarg(3); 
  if (ifarg(4)) { slowset=*getarg(4); delt=slowset/pg->cesz; slowset+=t; } 
  sf = gargstr(1);
  for (fi=0;fi<iflnum && strncmp(sf, &iflags[fi*4], 3)!=0;fi++) ; // find flag by name
  if (fi==iflnum) {printf("INTF6 ERR: %s not found as a flag (%s)\n",sf,iflags); hxe();}
  if (ifarg(2)) {
    if (hoc_is_double_arg(2)) {  // either set to all or just to this one
      val=(unsigned char)*getarg(2);
      if (slowset) { // set one and come back
        printf("NOT IMPLEMENTED\n"); // ****NOT IMPLEMENTED****
      } else if (setfl) { // set them all
        for (ix=0;ix<pg->cesz;ix++) { lop(pg->ce,ix); (&qp->type)[fi]=val; }
      } else {  // just set this one
        (&ip->type)[fi]=((fi>=iflneg)?(char)val:val);
      }
    } else {
      nx=vector_arg_px(2,&x);
      if (nx!=pg->cesz) {
        if (setfl) { printf("INTF6 flag ERR: vec sz mismatch: %d %d\n",nx,pg->cesz); hxe();
        } else       x=vector_newsize(vector_arg(2),pg->cesz);
      }
      if (setfl && slowset) { // set one and come back
        ix=0;
        lop(pg->ce,ix);
        (&qp->type)[fi]=((fi>=iflneg)?(char)x[ix]:(unsigned char)x[ix]);
        ix++;
        #if defined(t)
        net_send((void**)0x0, wts,tpnt,t+delt,OK); // OK is flag() flag
        #else
        net_send((void**)0x0, wts,tpnt,delt,OK);
        #endif
      } else for (ix=0;ix<pg->cesz;ix++) { 
        lop(pg->ce,ix); 
        if (setfl) { (&qp->type)[fi]=((fi>=iflneg)?(char)x[ix]:(unsigned char)x[ix]);
        } else {
          x[ix]=(double)((fi>=iflneg)?(char)(&qp->type)[fi]:(unsigned char)(&qp->type)[fi]);
        }
      }
    }
  }
  _lflag=(double)((fi>=iflneg)?(char)(&ip->type)[fi]:(unsigned char)(&ip->type)[fi]);
  ENDVERBATIM
}

FUNCTION allspck () {
  VERBATIM
  int i; double *x, sum; void *voi;
  ip = IDP; pg=ip->pg;
  voi=vector_arg(1);  x=vector_newsize(voi,pg->cesz);
  for (i=0,sum=0;i<pg->cesz;i++) { lopr(pg->ce,i); 
    x[i]=spck;
    sum+=spck;
  }
  _lallspck=sum;
  ENDVERBATIM
}

:** resetall()
PROCEDURE resetall () {
  VERBATIM
  int ii,i; unsigned char val;
  ip=IDP; pg=ip->pg;
  if(verbose>1) printf("resetall: ip=%p, col=%d, pg=%p\n",ip,pg->col,pg);
  for (i=0;i<pg->cesz;i++) { lopr(pg->ce,i);
    Vm=RMP; VAM=0; VNM=0; VGA=0; AHP=0; invlt=-1; VAM2=0; VNM2=0; VGA2=0;
    t0=t; trrs=t; twg = t; cbur=0; spck=0; refractory=0; VTHC=VTHR=VTH; 
  }
  ENDVERBATIM
}

:** floc(x,y[,z],vid,vdist,radius,type) // find cells within distance 'r' from a location 
: type can be a number to pick up or a vector to find all
FUNCTION floc () {
  VERBATIM
  double x,y,z,r,min,rad, *ix, *dd, *tdy; int ii,i,n,cnt,ty,tvf; void *voi, *vod, *voty;
  cnt=0; n=1000; r=0; z=ty=1e9; tvf=0;
  ip = IDP; pg=ip->pg;
  x = *getarg(1);
  y = *getarg(2);
  i=3;
  if (ifarg(i)) if (hoc_is_double_arg(i)) { z=*getarg(3); i++; }
  if (ifarg(i)) {
    voi=vector_arg(i++); ix=vector_newsize(voi,n); // id vector
    vod=vector_arg(i++); dd=vector_newsize(vod,n); // distance vector
    r= *getarg(i++); 
  }
  if (ifarg(i)) if (hoc_is_double_arg(i)) ty= *getarg(7); else { // type or -1 for EXCIT or -2 for INHIB
    tvf=1; voty=vector_arg(i++); tdy=vector_newsize(voty,n); // type vector
  } 
  r*=r; // squared for comparisons
  for (i=0,min=1e9,ii=-1;i<pg->cesz;i++) { qp=lopr(pg->ce,i); 
    if (ty!=1e9 && ((ty>=0 && ty!=qp->type) || (ty==-1 && qp->inhib==1) || (ty==-2 && qp->inhib==0))) continue;
    rad=(x-xloc)*(x-xloc)+(y-yloc)*(y-yloc)+(z==1e9?0.:((z-zloc)*(z-zloc))); // dist^2
    if (r>0 && rad<r) {
      // printf("AAAA: %d %g,%g,%g dist:%g (%g,%g,%g) mg0:%g\n",qp->id,xloc,yloc,zloc,sqrt(rad),x,y,z,mg0);
      if (cnt>=n) { // resize the vectors
        ix=vector_newsize(voi,n*=2); 
        dd=vector_newsize(vod,n);
        if (tvf) tdy=vector_newsize(voty,n);
      }
      ix[cnt]=(double)i;
      dd[cnt]=sqrt(rad); 
      if (tvf) tdy[cnt]=(double)qp->type;
      cnt++;
    } else if (rad<min) { min=rad; ii=i; }
  }
  if (r>0) { 
    ix=vector_newsize(voi,cnt); dd=vector_newsize(vod,cnt); 
    if (tvf) tdy=vector_newsize(voty,cnt);
    _lfloc=(double)cnt; } else {
    _lfloc=(double)ii;  } // return the index of the closest cell found
  ENDVERBATIM
}

:** invlset([val]) set or get the invl flag
FUNCTION invlset () {
  VERBATIM
  ip=IDP;
  if (ifarg(1)) ip->invl0 = (unsigned char) *getarg(1);
  _linvlset=(double)ip->invl0;
  ENDVERBATIM
}

:** vinset([val]) set or get the input flag (for using shared input from a vector)
FUNCTION vinset () {
  VERBATIM
  ip=IDP;
  if (ifarg(1)) ip->vinflg = (unsigned char) *getarg(1);
  if (ip->vinflg==1) {
    ip->input=1;
    ip->rvi = ip->rvb;
  }
  _lvinset=(double)ip->vinflg;
  ENDVERBATIM
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

PROCEDURE ESINo (x) {
  TABLE ESIN FROM 0 TO 2*PI WITH 3000 : one cycle
  ESIN = sin(x)
}

FUNCTION rates (vv) {
  : from Stevens & Jahr 1990a,b
  rates = maxnmc / (1 + exp(0.062 (/mV) * -vv) * ( (mg / mg0) ) )
}

: apply dopamine reward/punishment using pdope pointers, which store eligiblity signals as times
: of occurrence -- intf.dopelearn(1=='DA-burst' -1=='DA-dip') -- applies dopamine learning to all cells
: in same ce as calling INTF6. 
: rules:
:  DA-burst with pre-before-post causes LTP.
:  DA-burst with post-before-pre causes LTD.
:  DA-dip   with pre-before-post causes LTD.
:  DA-dip   with post-before-pre causes LTP.
FUNCTION dopelearn () {
  VERBATIM
  Point_process *pnnt;
  int i , iCell, pot;
  double tmp,maxw,inc,d,tau,pdopet;
  if(seadsetting!=3.) return 0.0; // seadsetting==3 for DOPE, must be set before network setup
  pot = (int) *getarg(1); // 
  ip=IDP; pg=ip->pg;
  for(iCell=0;iCell<pg->cesz;iCell++){
    lop(pg->ce,iCell);
    if(!qp->dvt)continue; //don't write empty pointers if no divergence
    for(i=0;i<qp->dvt;i++){
      pdopet = fabs(qp->pdope[i]); // fabs for backwards elig trace (post before pre has neg sign)
      d = t - pdopet; // time since eligibility trace turned on
      if(qp->pdope[i] > -1e9 && d <= maxeligtrdur ) { // -1e9 means it wasn't activated. maxeligtrdur is max time
        tmp = qp->wgain[i]; // current weight gain of synapse
        maxw = qp->pplastmaxw[i]; // max possible weight for the synapse
        tau = qp->pplasttau[i]; // time constant
        if( ! ( inc = qp->pplastinc[i] ) ) continue; // if plasticity is off at this synapse
        if(pot>0) { // DA-burst
          if(qp->pdope[i] >= 0) { // DA-burst with pre-before-post -> LTP
            if(SOFTSTDP) inc *= (1.0 - tmp / maxw); // soft bound for potentiation
            if (EXPELIGTR) // if we want to use exponential decay
              qp->wgain[i] += EPOTW * inc * exp( -d / tau ); // increment the wgain of the synapse
            else
              qp->wgain[i] += EPOTW * inc;
          } else { // DA-burst with post-before-pre -> LTD
            if(SOFTSTDP) inc *= (tmp / maxw); // soft bound for depression
            if (EXPELIGTR) // if we want to use exponential decay
              qp->wgain[i] -= EDEPW * inc * exp( -d / tau ); // increment the wgain of the synapse
            else
              qp->wgain[i] -= EDEPW * inc;
          }
        } else { // DA-dip
            if(qp->pdope[i] >= 0) { // DA-dip with pre-before-post -> LTD
              if(SOFTSTDP) inc *= (tmp / maxw); // soft bound for depression
              if (EXPELIGTR) // if we want to use exponential decay
                qp->wgain[i] -= EDEPW * inc * exp( -d / tau ); // increment the wgain of the synapse
              else
                qp->wgain[i] -= EDEPW * inc;
            } else { // DA-dip with post-before-pre -> LTP
              if(SOFTSTDP) inc *= (1.0 - tmp / maxw); // soft bound for potentiation
              if (EXPELIGTR) // if we want to use exponential decay
                qp->wgain[i] += EPOTW * inc * exp( -d / tau ); // increment the wgain of the synapse
              else
                qp->wgain[i] += EPOTW * inc;
            }
        }
        // check bounds of wgain
        if(qp->wgain[i]<0.) qp->wgain[i]=0.; else if(!SOFTSTDP && qp->wgain[i]>maxw) qp->wgain[i]=maxw;
        if(reseteligtr) qp->pdope[i] = -1e9; // reset here once synapse rewarded/punished
      }
    }
  }
  return 1.0;
  ENDVERBATIM
}

: intf.setdeletion - see comments below - used by homeostatic synaptic scaling
PROCEDURE setdeletion () {
  VERBATIM
  // Allow neurons to spontaneously die in proportion to their scaling factor (modified by
  // a rate constant which can be supplied as an argument).
  // This allows investigation into the progression of Alzheimer's disease as a via synaptic
  // scaling (Small, 2008).
  //
  // ARGUMENT: dynamic deletion rate constant (<=0 turns off dynamic deletion, >0 turns it on and
  // sets the rate constant).
  double x = *getarg(1);
  if (x <= 0) {
    dynamicdel = 0; // Turn off dynamic deletion
  } else {
    dynamicdel = 1; // Turn on dynamic deletion
    delspeed = x; // Set deletion rate constant
    printf("Set dynamic deletion rate constant = %e\n", delspeed);
  }
  ENDVERBATIM
}

VERBATIM
void dynamicdelete (double time) {
  // Allow this cell to die, with probability proportional to excitation and rate constant
  // Integrate over time since last call.
  
  // mcell_ran4 appears to take following args:
  // seed, pointer to RNG output store, num. random values to generate, range of random var (?)
  mcell_ran4(&sead, dscr, 1, 1.0); // Generate random value
  double p = dscr[0]; // Get the random value generated by the mcell_ran4 call

  double difference = ip->activity / ip->goal_activity; // Find magnitude of difference between activity and goal activity

  // Check if p < (difference-2)^2 * delspeed, normalised by time since last check, t - t'.
  //
  // This means that when difference = 2, chance of deletion is zero
  // (i.e. all cells are allowed to at least double their baseline firing rates without
  // risking excitotoxicity).
  // When difference = 3, chance of deletion per second is delspeed.
  // When difference = 4, chance of deletion per second is exponentially higher, etc.
  double x = difference - 2.0;
  if (x < 0) {
    x = 0; // Prevent scalefactors which are <1 from having a positive x^2
  }
  double threshold =  x * x * delspeed * ((time - ip->lastupdate) / 1000.0);

  if (p < threshold) {
    printf("p = %e, threshold = %e from x^2 * delspeed * timegap:\nx = %e, x^2 = %e, delspeed = %e, x^2*delspeed = %e, timegap (s) = %e\n", p, threshold, x, x*x, delspeed, x*x*delspeed, (time-ip->lastupdate)/1000.0);

    ip->dead = 1; // Kill cell
    printf("Cell %d has just died (scalefactor = %f)\n\n", ip->id, ip->scalefactor);
  }
}
ENDVERBATIM

VERBATIM
double get_avg_activity () {
  //Start by implementing retrospectively (i.e. assume network has been trained according to
  //pre-defined activity levels, but manually set targets retrospectively to these values by observation)
  // - with more time, implement Turrigiano (2008)'s "Factor+ vs Factor-" which balances firing rates
  
  // We could simply set goal_activity to be the current average activity value using
  //return ip->activity;
  // at an arbitrary time (e.g. when we turn on scaling). But as 'current' activity fluctuates, we don't
  // want to be stuck maintaining an unrealistic average firing rate, in the case that the 'current'
  // activity value was unusually high at the time of setting the goal.
  // It would be better to record an average of the neuron's activity so far, and use that as the goal.
  return ip->spkcnt / t;
}
ENDVERBATIM

VERBATIM 
void raise_activity_sensor (double time) {
  // Update the cell's activity sensor value, assuming this function has been called at the same
  // time as a spike at time t
  // REQUIRES: time of current spike in ms
  // ENSURES: returns activity value in MHz (due to ms timing)
  
  // Raise the activity by (-a + 1) / tau
  ip->activity = ip->activity + (-ip->activity + 1.0) / activitytau;

  // Update lastupdate time for next decay operation
  // -- probably shouldn't set this here, as it's also set in the NET_RECEIVE block on every
  // incoming event, and multiple updates of ip->lastupdate in different places will just lead
  // to confusion
  //ip->lastupdate = time;
  
  // DEBUG: (pick a random cell ID)
  //if (ip->id == 9 || ip->id == 10) {
  //  printf("spike from cell %d (inhib = %d) at time %f --> activity sensor = %f, target activity = %f, average activity = %f, scale = %f\n", ip->id, ip->inhib, time, ip->activity, ip->goal_activity, get_avg_activity(), ip->scalefactor);
  //}
}
ENDVERBATIM

VERBATIM
void decay_activity_sensor (double time) {
  // Decay the cell's activity sensor value according to the time since last decay update
  // In van Rossum et al. (2000), this is called every discrete timestep t
  // But this procedure is only called on NET_RECEIVE events, so we need to decay
  // taking into account the time since the last decay operation.

  // a_t = a_t0 * e(-(1/tau * t-t0))
  ip->activity = ip->activity * exp(-activityoneovertau * (time - ip->lastupdate));
}
ENDVERBATIM

VERBATIM
void update_scale_factor (double time) {
  // Implements weight scaling according to van Rossum et al. (2000)

  // Get difference between goal and current activity
  double err = ip->goal_activity - ip->activity;

  // Bound error to max_err value in the case that the activity sensor saturates during epileptic activity
  // This should prevent the integral from becoming excessively large over a relatively short time,
  // and therefore affecting the scaling for a very long time into the future.
  //if (err > ip->max_err) {
    //err = ip->max_err;
  //}
  //if (err < -ip->max_err) {
    //err = -ip->max_err;
  //}

  // Set scalefactor
  ip->scalefactor += (activitybeta * ip->scalefactor * err + activitygamma * ip->scalefactor * ip->activity_integral_err);

  // Bound scalefactor to max_scale to prevent Inf values
  if (ip->scalefactor > ip->max_scale) {
    ip->scalefactor = ip->max_scale;
  }

  // Calculate integral error term between sensor and target activity for next time (t')
  double timecorrection = time - ip->lastupdate;
  // e.g. If last update was 1ms ago, then the time correction = 1
  // If last update was 0.1ms ago correction = 0.1, so the accumulated error will be much smaller
  // If it's been a long time since the last update, the error will be correspondingly much larger

  ip->activity_integral_err += (err * timecorrection);
  // DEBUG: (pick a random cell ID)
  //if (ip->id == 9 || ip->id == 10) {
  //  printf("cell %d err = %f, time-corrected err = %f, integral_err = %f\n", ip->id, err, (err * timecorrection), ip->activity_integral_err);
  //}
}
ENDVERBATIM

: intf.scalefactor - optional arg sets value
:  used by homeostatic synaptic scaling
FUNCTION scalefactor () {
  VERBATIM
  if (ifarg(1)) IDP->scalefactor = *getarg(1);
  return IDP->scalefactor; // Return this cell's scale factor
  ENDVERBATIM
}

: intf.activity - optional arg sets value
:  used by homeostatic synaptic scaling
FUNCTION activity () {
  VERBATIM
  if (ifarg(1)) IDP->activity = *getarg(1);
  return IDP->activity; // Return this cell's activity sensor value
  ENDVERBATIM
}

: intf.goalactivity - optional arg sets value 
:  used by homeostatic synaptic scaling - target firing rate
FUNCTION goalactivity () {
  VERBATIM
  if (ifarg(1)) IDP->goal_activity = *getarg(1);
  return IDP->goal_activity;   // Return this cell's target activity value
  ENDVERBATIM
}

: intf.isdead - is the cell dead?
FUNCTION isdead() {
  VERBATIM
  return IDP->dead; // Return this cell's 'dead' flag
  ENDVERBATIM
}
