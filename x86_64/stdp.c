/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__STDP
#define _nrn_initial _nrn_initial__STDP
#define nrn_cur _nrn_cur__STDP
#define _nrn_current _nrn_current__STDP
#define nrn_jacob _nrn_jacob__STDP
#define nrn_state _nrn_state__STDP
#define _net_receive _net_receive__STDP 
#define adjustweight adjustweight__STDP 
#define reward_punish reward_punish__STDP 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define tauhebb _p[0]
#define tauanti _p[1]
#define hebbwt _p[2]
#define antiwt _p[3]
#define RLwindhebb _p[4]
#define RLwindanti _p[5]
#define useRLexp _p[6]
#define RLlenhebb _p[7]
#define RLlenanti _p[8]
#define RLhebbwt _p[9]
#define RLantiwt _p[10]
#define wmax _p[11]
#define softthresh _p[12]
#define STDPon _p[13]
#define RLon _p[14]
#define verbose _p[15]
#define skip _p[16]
#define tlastpre _p[17]
#define tlastpost _p[18]
#define tlasthebbelig _p[19]
#define tlastantielig _p[20]
#define interval _p[21]
#define deltaw _p[22]
#define newweight _p[23]
#define _tsav _p[24]
#define _nd_area  *_ppvar[0]._pval
#define synweight	*_ppvar[2]._pval
#define _p_synweight	_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  2;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_antiRL();
 static double _hoc_adjustweight();
 static double _hoc_hebbRL();
 static double _hoc_reward_punish();
 static double _hoc_softthreshold();
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(_ho) Object* _ho; { void* create_point_process();
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt();
 static double _hoc_loc_pnt(_vptr) void* _vptr; {double loc_point_process();
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(_vptr) void* _vptr; {double has_loc_point();
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(_vptr)void* _vptr; {
 double get_loc_point_process(); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "antiRL", _hoc_antiRL,
 "adjustweight", _hoc_adjustweight,
 "hebbRL", _hoc_hebbRL,
 "reward_punish", _hoc_reward_punish,
 "softthreshold", _hoc_softthreshold,
 0, 0
};
#define antiRL antiRL_STDP
#define hebbRL hebbRL_STDP
#define softthreshold softthreshold_STDP
 extern double antiRL( );
 extern double hebbRL( );
 extern double softthreshold( double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tauhebb", "ms",
 "tauanti", "ms",
 "RLwindhebb", "ms",
 "RLwindanti", "ms",
 "RLlenhebb", "ms",
 "RLlenanti", "ms",
 "tlastpre", "ms",
 "tlastpost", "ms",
 "tlasthebbelig", "ms",
 "tlastantielig", "ms",
 "interval", "ms",
 0,0
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(_vptr) void* _vptr; {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"STDP",
 "tauhebb",
 "tauanti",
 "hebbwt",
 "antiwt",
 "RLwindhebb",
 "RLwindanti",
 "useRLexp",
 "RLlenhebb",
 "RLlenanti",
 "RLhebbwt",
 "RLantiwt",
 "wmax",
 "softthresh",
 "STDPon",
 "RLon",
 "verbose",
 "skip",
 0,
 "tlastpre",
 "tlastpost",
 "tlasthebbelig",
 "tlastantielig",
 "interval",
 "deltaw",
 "newweight",
 0,
 0,
 "synweight",
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 25, _prop);
 	/*initialize range parameters*/
 	tauhebb = 10;
 	tauanti = 10;
 	hebbwt = 1;
 	antiwt = -1;
 	RLwindhebb = 10;
 	RLwindanti = 10;
 	useRLexp = 0;
 	RLlenhebb = 100;
 	RLlenanti = 100;
 	RLhebbwt = 1;
 	RLantiwt = -1;
 	wmax = 15;
 	softthresh = 0;
 	STDPon = 1;
 	RLon = 1;
 	verbose = 0;
 	skip = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 25;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[3]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _stdp_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,(void*)0, (void*)0, (void*)0, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 25, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
  hoc_register_dparam_semantics(_mechtype, 3, "netsend");
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 STDP /Users/anwarharoon/Documents/NKI-modeling/SMARTAgent/x86_64/stdp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int adjustweight(double);
static int reward_punish(double);
 
static void _net_receive (_pnt, _args, _lflag) Point_process* _pnt; double* _args; double _lflag; 
{    _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   deltaw = 0.0 ;
   skip = 0.0 ;
   if ( verbose > 0.0 ) {
     printf ( "t=%f (BEFORE) tlaspre=%f, tlastpost=%f, flag=%f, w=%f, deltaw=%f \n" , t , tlastpre , tlastpost , _lflag , _args[0] , deltaw ) ;
     }
   if ( ( _lflag  == - 1.0 )  && ( tlastpre  != t - 1.0 ) ) {
     skip = 1.0 ;
     deltaw = hebbwt * exp ( - interval / tauhebb ) ;
     if ( softthresh  == 1.0 ) {
       deltaw = softthreshold ( _threadargscomma_ deltaw ) ;
       }
     adjustweight ( _threadargscomma_ deltaw ) ;
     if ( verbose > 0.0 ) {
       printf ( "Hebbian STDP event: t = %f ms; tlastpre = %f; w = %f; deltaw = %f\n" , t , tlastpre , _args[0] , deltaw ) ;
       }
     }
   else if ( ( _lflag  == 1.0 )  && ( tlastpost  != t - 1.0 ) ) {
     skip = 1.0 ;
     deltaw = antiwt * exp ( interval / tauanti ) ;
     if ( softthresh  == 1.0 ) {
       deltaw = softthreshold ( _threadargscomma_ deltaw ) ;
       }
     adjustweight ( _threadargscomma_ deltaw ) ;
     if ( verbose > 0.0 ) {
       printf ( "anti-Hebbian STDP event: t = %f ms; deltaw = %f\n" , t , deltaw ) ;
       }
     }
   if ( skip  == 0.0 ) {
     if ( _args[0] >= 0.0 ) {
       interval = tlastpost - t ;
       if ( ( tlastpost > - 1.0 )  && ( - interval > 1.0 ) ) {
         if ( STDPon  == 1.0 ) {
           if ( verbose > 0.0 ) {
             printf ( "net_send(1,1)\n" ) ;
             }
           net_send ( _tqitem, _args, _pnt, t +  1.0 , 1.0 ) ;
           }
         if ( ( RLon  == 1.0 )  && ( - interval <= RLwindanti ) ) {
           tlastantielig = t ;
           }
         }
       tlastpre = t ;
       }
     else {
       interval = t - tlastpre ;
       if ( ( tlastpre > - 1.0 )  && ( interval > 1.0 ) ) {
         if ( STDPon  == 1.0 ) {
           if ( verbose > 0.0 ) {
             printf ( "net_send(1,-1)\n" ) ;
             }
           net_send ( _tqitem, _args, _pnt, t +  1.0 , - 1.0 ) ;
           }
         if ( ( RLon  == 1.0 )  && ( interval <= RLwindhebb ) ) {
           tlasthebbelig = t ;
           }
         }
       tlastpost = t ;
       }
     }
   if ( verbose > 0.0 ) {
     printf ( "t=%f (AFTER) tlaspre=%f, tlastpost=%f, flag=%f, w=%f, deltaw=%f \n" , t , tlastpre , tlastpost , _lflag , _args[0] , deltaw ) ;
     }
   } }
 
static int  reward_punish (  double _lreinf ) {
   if ( RLon  == 1.0 ) {
     deltaw = 0.0 ;
     deltaw = deltaw + _lreinf * hebbRL ( _threadargs_ ) ;
     deltaw = deltaw + _lreinf * antiRL ( _threadargs_ ) ;
     if ( softthresh  == 1.0 ) {
       deltaw = softthreshold ( _threadargscomma_ deltaw ) ;
       }
     adjustweight ( _threadargscomma_ deltaw ) ;
     if ( verbose > 0.0 ) {
       printf ( "RL event: t = %f ms; reinf = %f; RLhebbwt = %f; RLlenhebb = %f; tlasthebbelig = %f; deltaw = %f\n" , t , _lreinf , RLhebbwt , RLlenhebb , tlasthebbelig , deltaw ) ;
       }
     }
    return 0; }
 
static double _hoc_reward_punish(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 reward_punish (  *getarg(1) );
 return(_r);
}
 
double hebbRL (  ) {
   double _lhebbRL;
 if ( ( RLon  == 0.0 )  || ( tlasthebbelig < 0.0 ) ) {
     _lhebbRL = 0.0 ;
     }
   else if ( useRLexp  == 0.0 ) {
     if ( t - tlasthebbelig <= RLlenhebb ) {
       _lhebbRL = RLhebbwt ;
       }
     else {
       _lhebbRL = 0.0 ;
       }
     }
   else {
     _lhebbRL = RLhebbwt * exp ( ( tlasthebbelig - t ) / RLlenhebb ) ;
     }
   
return _lhebbRL;
 }
 
static double _hoc_hebbRL(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  hebbRL (  );
 return(_r);
}
 
double antiRL (  ) {
   double _lantiRL;
 if ( ( RLon  == 0.0 )  || ( tlastantielig < 0.0 ) ) {
     _lantiRL = 0.0 ;
     }
   else if ( useRLexp  == 0.0 ) {
     if ( t - tlastantielig <= RLlenanti ) {
       _lantiRL = RLantiwt ;
       }
     else {
       _lantiRL = 0.0 ;
       }
     }
   else {
     _lantiRL = RLantiwt * exp ( ( tlastantielig - t ) / RLlenanti ) ;
     }
   
return _lantiRL;
 }
 
static double _hoc_antiRL(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  antiRL (  );
 return(_r);
}
 
double softthreshold (  double _lrawwc ) {
   double _lsoftthreshold;
 if ( _lrawwc >= 0.0 ) {
     _lsoftthreshold = _lrawwc * ( 1.0 - synweight / wmax ) ;
     }
   else {
     _lsoftthreshold = _lrawwc * synweight / wmax ;
     }
   
return _lsoftthreshold;
 }
 
static double _hoc_softthreshold(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  softthreshold (  *getarg(1) );
 return(_r);
}
 
static int  adjustweight (  double _lwc ) {
   synweight = synweight + _lwc ;
   if ( synweight > wmax ) {
     synweight = wmax ;
     }
   if ( synweight < 0.0 ) {
     synweight = 0.0 ;
     }
    return 0; }
 
static double _hoc_adjustweight(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 adjustweight (  *getarg(1) );
 return(_r);
}

static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   tlastpre = - 1.0 ;
   tlastpost = - 1.0 ;
   tlasthebbelig = - 1.0 ;
   tlastantielig = - 1.0 ;
   interval = 0.0 ;
   deltaw = 0.0 ;
   newweight = 0.0 ;
   }

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/anwarharoon/Documents/NKI-modeling/SMARTAgent/stdp.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "\n"
  "STDP + RL weight adjuster mechanism\n"
  "\n"
  "Original STDP code adapted from:\n"
  "http://senselab.med.yale.edu/modeldb/showmodel.asp?model=64261&file=\\bfstdp\\stdwa_songabbott.mod\n"
  "\n"
  "Adapted to implement a \"nearest-neighbor spike-interaction\" model (see \n"
  "Scholarpedia article on STDP) that just looks at the last-seen pre- and \n"
  "post-synaptic spikes, and implementing a reinforcement learning algorithm based\n"
  "on (Chadderdon et al., 2012):\n"
  "http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0047251\n"
  "\n"
  "Example Python usage:\n"
  "\n"
  "from neuron import h\n"
  "\n"
  "## Create cells\n"
  "dummy = h.Section() # Create a dummy section to put the point processes in\n"
  "ncells = 2\n"
  "cells = []\n"
  "for c in range(ncells): cells.append(h.IntFire4(0,sec=dummy)) # Create the cells\n"
  "\n"
  "## Create synapses\n"
  "threshold = 10 # Set voltage threshold\n"
  "delay = 1 # Set connection delay\n"
  "singlesyn = h.NetCon(cells[0],cells[1], threshold, delay, 0.5) # Create a connection between the cells\n"
  "stdpmech = h.STDP(0,sec=dummy) # Create the STDP mechanism\n"
  "presyn = h.NetCon(cells[0],stdpmech, threshold, delay, 1) # Feed presynaptic spikes to the STDP mechanism -- must have weight >0\n"
  "pstsyn = h.NetCon(cells[1],stdpmech, threshold, delay, -1) # Feed postsynaptic spikes to the STDP mechanism -- must have weight <0\n"
  "h.setpointer(singlesyn._ref_weight[0],'synweight',stdpmech) # Point the STDP mechanism to the connection weight\n"
  "\n"
  "Version: 2013oct24 by cliffk\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "    POINT_PROCESS STDP : Definition of mechanism\n"
  "    POINTER synweight : Pointer to the weight (in a NetCon object) to be adjusted.\n"
  "    RANGE tauhebb, tauanti : LTP/LTD decay time constants (in ms) for the Hebbian (pre-before-post-synaptic spikes), and anti-Hebbian (post-before-pre-synaptic) cases. \n"
  "    RANGE hebbwt, antiwt : Maximal adjustment (can be positive or negative) for Hebbian and anti-Hebbian cases (i.e., as inter-spike interval approaches zero).  This should be set positive for LTP and negative for LTD.\n"
  "    RANGE RLwindhebb, RLwindanti : Maximum interval between pre- and post-synaptic events for an starting an eligibility trace.  There are separate ones for the Hebbian and anti-Hebbian events.\n"
  "    RANGE useRLexp : Use exponentially decaying eligibility traces?  If 0, then the eligibility traces are binary, turning on at the beginning and completely off after time has passed corresponding to RLlen.\n"
  "    RANGE RLlenhebb, RLlenanti : Length of the eligibility Hebbian and anti-Hebbian eligibility traces, or the decay time constants if the traces are decaying exponentials.\n"
  "    RANGE RLhebbwt, RLantiwt : Maximum gains to be applied to the reward or punishing signal by Hebbian and anti-Hebbian eligibility traces.  \n"
  "    RANGE wmax : The maximum weight for the synapse.\n"
  "    RANGE softthresh : Flag turning on \"soft thresholding\" for the maximal adjustment parameters.\n"
  "    RANGE STDPon : Flag for turning STDP adjustment on / off.\n"
  "    RANGE RLon : Flag for turning RL adjustment on / off.\n"
  "    RANGE verbose : Flag for turning off prints of weight update events for debugging.\n"
  "    RANGE tlastpre, tlastpost : Remembered times for last pre- and post-synaptic spikes.\n"
  "    RANGE tlasthebbelig, tlastantielig : Remembered times for Hebbian anti-Hebbian eligibility traces.\n"
  "    RANGE interval : Interval between current time t and previous spike.\n"
  "    RANGE deltaw : The calculated weight change.\n"
  "    RANGE newweight : New calculated weight.\n"
  "    RANGE skip : Flag to skip 2nd set of conditions\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    synweight        \n"
  "    tlastpre   (ms)    \n"
  "    tlastpost  (ms)   \n"
  "    tlasthebbelig   (ms)    \n"
  "    tlastantielig  (ms)        \n"
  "    interval    (ms)    \n"
  "    deltaw\n"
  "    newweight          \n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    tlastpre = -1            : no spike yet\n"
  "    tlastpost = -1           : no spike yet\n"
  "    tlasthebbelig = -1      : no eligibility yet\n"
  "    tlastantielig = -1  : no eligibility yet   \n"
  "    interval = 0\n"
  "    deltaw = 0\n"
  "    newweight = 0\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    tauhebb  = 10  (ms)   \n"
  "    tauanti  = 10  (ms)    \n"
  "    hebbwt = 1.0\n"
  "    antiwt = -1.0\n"
  "    RLwindhebb = 10 (ms)\n"
  "    RLwindanti = 10 (ms)\n"
  "    useRLexp = 0   : default to using binary eligibility traces\n"
  "    RLlenhebb = 100 (ms)\n"
  "    RLlenanti = 100 (ms)\n"
  "    RLhebbwt = 1.0\n"
  "    RLantiwt = -1.0\n"
  "    wmax  = 15.0\n"
  "    softthresh = 0\n"
  "    STDPon = 1\n"
  "    RLon = 1\n"
  "    verbose = 0\n"
  "    skip = 0\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "    deltaw = 0.0 : Default the weight change to 0.\n"
  "    skip = 0\n"
  "    \n"
  "    if (verbose > 0)  { printf(\"t=%f (BEFORE) tlaspre=%f, tlastpost=%f, flag=%f, w=%f, deltaw=%f \\n\",t,tlastpre, tlastpost,flag,w,deltaw) }\n"
  "\n"
  "    : Hebbian weight update happens 1ms later to check for simultaneous spikes (otherwise bug when using mpi)\n"
  "    if ((flag == -1) && (tlastpre != t-1)) {   \n"
  "        skip = 1 : skip the 2nd set of conditions since this was artificial net event to update weights\n"
  "        deltaw = hebbwt * exp(-interval / tauhebb) : Use the Hebbian decay to set the Hebbian weight adjustment. \n"
  "        if (softthresh == 1) { deltaw = softthreshold(deltaw) } : If we have soft-thresholding on, apply it.\n"
  "        adjustweight(deltaw) : Adjust the weight.\n"
  "        if (verbose > 0) { printf(\"Hebbian STDP event: t = %f ms; tlastpre = %f; w = %f; deltaw = %f\\n\",t,tlastpre,w,deltaw) } : Show weight update information if debugging on.\n"
  "        }\n"
  "\n"
  "    : Ant-hebbian weight update happens 1ms later to check for simultaneous spikes (otherwise bug when using mpi)\n"
  "    else if ((flag == 1) && (tlastpost != t-1)) { :update weight 1ms later to check for simultaneous spikes (otherwise bug when using mpi)\n"
  "        skip = 1 : skip the 2nd set of conditions since this was artificial net event to update weights\n"
  "        deltaw = antiwt * exp(interval / tauanti) : Use the anti-Hebbian decay to set the anti-Hebbian weight adjustment.\n"
  "        if (softthresh == 1) { deltaw = softthreshold(deltaw) } : If we have soft-thresholding on, apply it.\n"
  "        adjustweight(deltaw) : Adjust the weight.\n"
  "        if (verbose > 0) { printf(\"anti-Hebbian STDP event: t = %f ms; deltaw = %f\\n\",t,deltaw) } : Show weight update information if debugging on. \n"
  "        }\n"
  "\n"
  "\n"
  "    : If we receive a non-negative weight value, we are receiving a pre-synaptic spike (and thus need to check for an anti-Hebbian event, since the post-synaptic weight must be earlier).\n"
  "    if (skip == 0) {\n"
  "        if (w >= 0) {           \n"
  "            interval = tlastpost - t  : Get the interval; interval is negative\n"
  "            if  ((tlastpost > -1) && (-interval > 1.0)) { : If we had a post-synaptic spike and a non-zero interval...\n"
  "                if (STDPon == 1) { : If STDP learning is turned on...\n"
  "                    if (verbose > 0) {printf(\"net_send(1,1)\\n\")}\n"
  "                    net_send(1,1) : instead of updating weight directly, use net_send to check if simultaneous spike occurred (otherwise bug when using mpi)\n"
  "                }\n"
  "                if ((RLon == 1) && (-interval <= RLwindanti)) { tlastantielig = t } : If RL and anti-Hebbian eligibility traces are turned on, and the interval falls within the maximum window for eligibility, remember the eligibilty trace start at the current time.\n"
  "            }\n"
  "            tlastpre = t : Remember the current spike time for next NET_RECEIVE.  \n"
  "        \n"
  "        : Else, if we receive a negative weight value, we are receiving a post-synaptic spike (and thus need to check for a Hebbian event, since the pre-synaptic weight must be earlier).    \n"
  "        } else {            \n"
  "            interval = t - tlastpre : Get the interval; interval is positive\n"
  "            if  ((tlastpre > -1) && (interval > 1.0)) { : If we had a pre-synaptic spike and a non-zero interval...\n"
  "                if (STDPon == 1) { : If STDP learning is turned on...\n"
  "                    if (verbose > 0) {printf(\"net_send(1,-1)\\n\")}\n"
  "                    net_send(1,-1) : instead of updating weight directly, use net_send to check if simultaneous spike occurred (otherwise bug when using mpi)\n"
  "                }\n"
  "                if ((RLon == 1) && (interval <= RLwindhebb)) { \n"
  "                    tlasthebbelig = t} : If RL and Hebbian eligibility traces are turned on, and the interval falls within the maximum window for eligibility, remember the eligibilty trace start at the current time.\n"
  "            }\n"
  "            tlastpost = t : Remember the current spike time for next NET_RECEIVE.\n"
  "        }\n"
  "    }\n"
  "    if (verbose > 0)  { printf(\"t=%f (AFTER) tlaspre=%f, tlastpost=%f, flag=%f, w=%f, deltaw=%f \\n\",t,tlastpre, tlastpost,flag,w,deltaw) }\n"
  "}\n"
  "\n"
  "PROCEDURE reward_punish(reinf) {\n"
  "    if (RLon == 1) { : If RL is turned on...\n"
  "        deltaw = 0.0 : Start the weight change as being 0.\n"
  "        deltaw = deltaw + reinf * hebbRL() : If we have the Hebbian eligibility traces on, add their effect in.   \n"
  "        deltaw = deltaw + reinf * antiRL() : If we have the anti-Hebbian eligibility traces on, add their effect in.\n"
  "        if (softthresh == 1) { deltaw = softthreshold(deltaw) }  : If we have soft-thresholding on, apply it.  \n"
  "        adjustweight(deltaw) : Adjust the weight.\n"
  "        if (verbose > 0) { printf(\"RL event: t = %f ms; reinf = %f; RLhebbwt = %f; RLlenhebb = %f; tlasthebbelig = %f; deltaw = %f\\n\",t,reinf,RLhebbwt,RLlenhebb,tlasthebbelig, deltaw) } : Show weight update information if debugging on.     \n"
  "    }\n"
  "}\n"
  "\n"
  "FUNCTION hebbRL() {\n"
  "    if ((RLon == 0) || (tlasthebbelig < 0.0)) { hebbRL = 0.0  } : If RL is turned off or eligibility has not occurred yet, return 0.0.\n"
  "    else if (useRLexp == 0) { : If we are using a binary (i.e. square-wave) eligibility traces...\n"
  "        if (t - tlasthebbelig <= RLlenhebb) { hebbRL = RLhebbwt } : If we are within the length of the eligibility trace...\n"
  "        else { hebbRL = 0.0 } : Otherwise (outside the length), return 0.0.\n"
  "    } \n"
  "    else { hebbRL = RLhebbwt * exp((tlasthebbelig - t) / RLlenhebb) } : Otherwise (if we're using an exponential decay traces)...use the Hebbian decay to calculate the gain.\n"
  "      \n"
  "}\n"
  "FUNCTION antiRL() {\n"
  "    if ((RLon == 0) || (tlastantielig < 0.0)) { antiRL = 0.0 } : If RL is turned off or eligibility has not occurred yet, return 0.0.\n"
  "    else if (useRLexp == 0) { : If we are using a binary (i.e. square-wave) eligibility traces...\n"
  "        if (t - tlastantielig <= RLlenanti) { antiRL = RLantiwt } : If we are within the length of the eligibility trace...\n"
  "        else {antiRL = 0.0 } : Otherwise (outside the length), return 0.0.\n"
  "    }\n"
  "    else { antiRL = RLantiwt * exp((tlastantielig - t) / RLlenanti) } : Otherwise (if we're using an exponential decay traces), use the anti-Hebbian decay to calculate the gain.  \n"
  "}\n"
  "\n"
  "FUNCTION softthreshold(rawwc) {\n"
  "    if (rawwc >= 0) { softthreshold = rawwc * (1.0 - synweight / wmax) } : If the weight change is non-negative, scale by 1 - weight / wmax.\n"
  "    else { softthreshold = rawwc * synweight / wmax } : Otherwise (the weight change is negative), scale by weight / wmax.    \n"
  "}\n"
  "\n"
  "PROCEDURE adjustweight(wc) {\n"
  "   synweight = synweight + wc : apply the synaptic modification, and then clip the weight if necessary to make sure it's between 0 and wmax.\n"
  "   if (synweight > wmax) { synweight = wmax }\n"
  "   if (synweight < 0) { synweight = 0 }\n"
  "}\n"
  ;
#endif
