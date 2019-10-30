/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
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
 
#define nrn_init _nrn_init__NSLOC
#define _nrn_initial _nrn_initial__NSLOC
#define nrn_cur _nrn_cur__NSLOC
#define _nrn_current _nrn_current__NSLOC
#define nrn_jacob _nrn_jacob__NSLOC
#define nrn_state _nrn_state__NSLOC
#define _net_receive _net_receive__NSLOC 
#define init_sequence init_sequence__NSLOC 
#define next_invl next_invl__NSLOC 
#define noiseFromRandom noiseFromRandom__NSLOC 
#define seed seed__NSLOC 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define interval _p[0]
#define number _p[1]
#define start _p[2]
#define noise _p[3]
#define xloc _p[4]
#define yloc _p[5]
#define zloc _p[6]
#define id _p[7]
#define type _p[8]
#define subtype _p[9]
#define fflag _p[10]
#define event _p[11]
#define last_interval _p[12]
#define transition _p[13]
#define on _p[14]
#define ispike _p[15]
#define v _p[16]
#define _tsav _p[17]
#define _nd_area  *_ppvar[0]._pval
#define donotuse	*_ppvar[2]._pval
#define _p_donotuse	_ppvar[2]._pval
 
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_erand();
 static double _hoc_init_sequence();
 static double _hoc_invl();
 static double _hoc_next_invl();
 static double _hoc_noiseFromRandom();
 static double _hoc_seed();
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
 _extcall_prop = _prop;
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
 "erand", _hoc_erand,
 "init_sequence", _hoc_init_sequence,
 "invl", _hoc_invl,
 "next_invl", _hoc_next_invl,
 "noiseFromRandom", _hoc_noiseFromRandom,
 "seed", _hoc_seed,
 0, 0
};
#define erand erand_NSLOC
#define invl invl_NSLOC
 extern double erand( _threadargsproto_ );
 extern double invl( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define check_interval check_interval_NSLOC
 double check_interval = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "interval", 1e-09, 1e+09,
 "noise", 0, 1,
 "number", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "check_interval_NSLOC", "ms",
 "interval", "ms",
 "start", "ms",
 0,0
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "check_interval_NSLOC", &check_interval_NSLOC,
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
"NSLOC",
 "interval",
 "number",
 "start",
 "noise",
 "xloc",
 "yloc",
 "zloc",
 "id",
 "type",
 "subtype",
 "fflag",
 0,
 0,
 0,
 "donotuse",
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
 	_p = nrn_prop_data_alloc(_mechtype, 18, _prop);
 	/*initialize range parameters*/
 	interval = 100;
 	number = 3000;
 	start = 1;
 	noise = 0;
 	xloc = -1;
 	yloc = -1;
 	zloc = -1;
 	id = -1;
 	type = -1;
 	subtype = -1;
 	fflag = 1;
  }
 	_prop->param = _p;
 	_prop->param_size = 18;
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

 void _nsloc_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,(void*)0, (void*)0, (void*)0, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 18, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
  hoc_register_dparam_semantics(_mechtype, 3, "netsend");
 add_nrn_artcell(_mechtype, 3);
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 NSLOC /Users/anwarharoon/Documents/NKI-modeling/SMARTAgent/x86_64/nsloc.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int init_sequence(_threadargsprotocomma_ double);
static int next_invl(_threadargsproto_);
static int noiseFromRandom(_threadargsproto_);
static int seed(_threadargsprotocomma_ double);
 
static int  seed ( _threadargsprotocomma_ double _lx ) {
   set_seed ( _lx ) ;
    return 0; }
 
static double _hoc_seed(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (_NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 seed ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static int  init_sequence ( _threadargsprotocomma_ double _lt ) {
   if ( number > 0.0 ) {
     on = 1.0 ;
     event = 0.0 ;
     ispike = 0.0 ;
     }
    return 0; }
 
static double _hoc_init_sequence(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (_NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 init_sequence ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
double invl ( _threadargsprotocomma_ double _lmean ) {
   double _linvl;
 if ( _lmean <= 0. ) {
     _lmean = .01 ;
     }
   if ( noise  == 0.0 ) {
     _linvl = _lmean ;
     }
   else {
     _linvl = ( 1. - noise ) * _lmean + noise * _lmean * erand ( _threadargs_ ) ;
     }
   
return _linvl;
 }
 
static double _hoc_invl(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (_NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  invl ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
/*VERBATIM*/
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
 
double erand ( _threadargsproto_ ) {
   double _lerand;
 
/*VERBATIM*/
	if (_p_donotuse) {
		/*
		:Supports separate independent but reproducible streams for
		: each instance. However, the corresponding hoc Random
		: distribution MUST be set to Random.negexp(1)
		*/
		_lerand = nrn_random_pick(_p_donotuse);
	}else{
		/* only can be used in main thread */
		if (_nt != nrn_threads) {
hoc_execerror("multithread random in NetStim"," only via hoc Random");
		}
 _lerand = exprand ( 1.0 ) ;
   
/*VERBATIM*/
	}
 
return _lerand;
 }
 
static double _hoc_erand(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (_NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  erand ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int  noiseFromRandom ( _threadargsproto_ ) {
   
/*VERBATIM*/
 {
	void** pv = (void**)(&_p_donotuse);
	if (ifarg(1)) {
		*pv = nrn_random_arg(1);
	}else{
		*pv = (void*)0;
	}
 }
  return 0; }
 
static double _hoc_noiseFromRandom(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (_NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 noiseFromRandom ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int  next_invl ( _threadargsproto_ ) {
   if ( number > 0.0 ) {
     event = invl ( _threadargscomma_ interval ) ;
     }
   if ( ispike >= number ) {
     on = 0.0 ;
     }
    return 0; }
 
static double _hoc_next_invl(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (_NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 next_invl ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static void _net_receive (_pnt, _args, _lflag) Point_process* _pnt; double* _args; double _lflag; 
{  double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   _thread = (Datum*)0; _nt = (_NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 0.0 ) {
     if ( _args[0] > 0.0  && on  == 0.0 ) {
       init_sequence ( _threadargscomma_ t ) ;
       next_invl ( _threadargs_ ) ;
       event = event - interval * ( 1. - noise ) ;
       artcell_net_send ( _tqitem, _args, _pnt, t +  event , 1.0 ) ;
       }
     else if ( _args[0] < 0.0 ) {
       on = 0.0 ;
       }
     }
   if ( _lflag  == 3.0 ) {
     if ( on  == 1.0 ) {
       init_sequence ( _threadargscomma_ t ) ;
       artcell_net_send ( _tqitem, _args, _pnt, t +  0.0 , 1.0 ) ;
       artcell_net_send ( _tqitem, _args, _pnt, t +  2.0 * check_interval , 4.0 ) ;
       }
     }
   if ( _lflag  == 1.0  && on  == 1.0 ) {
     ispike = ispike + 1.0 ;
     net_event ( _pnt, t ) ;
     next_invl ( _threadargs_ ) ;
     transition = 0.0 ;
     if ( on  == 1.0 ) {
       artcell_net_send ( _tqitem, _args, _pnt, t +  event , 1.0 ) ;
       }
     }
   if ( _lflag  == 5.0  && on  == 1.0  && transition  == 1.0 ) {
     ispike = ispike + 1.0 ;
     net_event ( _pnt, t ) ;
     next_invl ( _threadargs_ ) ;
     if ( on  == 1.0 ) {
       artcell_net_send ( _tqitem, _args, _pnt, t +  event , 5.0 ) ;
       }
     }
   if ( _lflag  == 4.0  && on  == 1.0 ) {
     if ( interval < last_interval ) {
       next_invl ( _threadargs_ ) ;
       transition = 1.0 ;
       artcell_net_send ( _tqitem, _args, _pnt, t +  event , 5.0 ) ;
       }
     last_interval = interval ;
     artcell_net_send ( _tqitem, _args, _pnt, t +  check_interval , 4.0 ) ;
     }
   if ( _lflag  == 1.0  && on  == 0.0 ) {
     net_event ( _pnt, t ) ;
     }
   } }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
 {
   on = 0.0 ;
   ispike = 0.0 ;
   if ( noise < 0.0 ) {
     noise = 0.0 ;
     }
   if ( noise > 1.0 ) {
     noise = 1.0 ;
     }
   if ( start >= 0.0  && number > 0.0 ) {
     on = 1.0 ;
     event = start + invl ( _threadargscomma_ interval ) - interval * ( 1. - noise ) ;
     if ( event < 0.0 ) {
       event = 0.0 ;
       }
     artcell_net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  event , 3.0 ) ;
     }
   }

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
 v=_v;
{
}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/anwarharoon/Documents/NKI-modeling/SMARTAgent/nsloc.mod";
static const char* nmodl_file_text = 
  ": $Id: nsloc.mod,v 1.7 2013/06/20  salvad $\n"
  ": from nrn/src/nrnoc/netstim.mod\n"
  ": modified to use as proprioceptive units in arm2dms model\n"
  "\n"
  "NEURON	{ \n"
  "  ARTIFICIAL_CELL NSLOC\n"
  "  RANGE interval, number, start, xloc, yloc, zloc, id, type, subtype, fflag, mlenmin, mlenmax, checkInterval\n"
  "  RANGE noise\n"
  "  THREADSAFE : only true if every instance has its own distinct Random\n"
  "  POINTER donotuse\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	interval	= 100 (ms) <1e-9,1e9>: time between spikes (msec)\n"
  "	number	= 3000 <0,1e9>	: number of spikes (independent of noise)\n"
  "	start		= 1 (ms)	: start of first spike\n"
  "	noise		= 0 <0,1>	: amount of randomness (0.0 - 1.0)\n"
  "    xloc = -1 \n"
  "    yloc = -1 \n"
  "    zloc = -1         : location\n"
  "    id = -1\n"
  "    type = -1\n"
  "    subtype = -1\n"
  "    fflag           = 1             : don't change -- indicates that this is an artcell\n"
  "    check_interval = 1.0 (ms) : time between checking if interval has changed\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	event (ms)\n"
  "	last_interval (ms)\n"
  "	transition\n"
  "	on\n"
  "	ispike\n"
  "	donotuse\n"
  "}\n"
  "\n"
  "PROCEDURE seed(x) {\n"
  "	set_seed(x)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	on = 0 : off\n"
  "	ispike = 0\n"
  "	if (noise < 0) {\n"
  "		noise = 0\n"
  "	}\n"
  "	if (noise > 1) {\n"
  "		noise = 1\n"
  "	}\n"
  "	if (start >= 0 && number > 0) {\n"
  "		on = 1\n"
  "		: randomize the first spike so on average it occurs at\n"
  "		: start + noise*interval\n"
  "		event = start + invl(interval) - interval*(1. - noise)\n"
  "		: but not earlier than 0\n"
  "		if (event < 0) {\n"
  "			event = 0\n"
  "		}\n"
  "		net_send(event, 3)\n"
  "	}\n"
  "}	\n"
  "\n"
  "PROCEDURE init_sequence(t(ms)) {\n"
  "	if (number > 0) {\n"
  "		on = 1\n"
  "		event = 0\n"
  "		ispike = 0\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION invl(mean (ms)) (ms) {\n"
  "	if (mean <= 0.) {\n"
  "		mean = .01 (ms) : I would worry if it were 0.\n"
  "	}\n"
  "	if (noise == 0) {\n"
  "		invl = mean\n"
  "	}else{\n"
  "		invl = (1. - noise)*mean + noise*mean*erand()\n"
  "	}\n"
  "}\n"
  "VERBATIM\n"
  "double nrn_random_pick(void* r);\n"
  "void* nrn_random_arg(int argpos);\n"
  "ENDVERBATIM\n"
  "\n"
  "FUNCTION erand() {\n"
  "VERBATIM\n"
  "	if (_p_donotuse) {\n"
  "		/*\n"
  "		:Supports separate independent but reproducible streams for\n"
  "		: each instance. However, the corresponding hoc Random\n"
  "		: distribution MUST be set to Random.negexp(1)\n"
  "		*/\n"
  "		_lerand = nrn_random_pick(_p_donotuse);\n"
  "	}else{\n"
  "		/* only can be used in main thread */\n"
  "		if (_nt != nrn_threads) {\n"
  "hoc_execerror(\"multithread random in NetStim\",\" only via hoc Random\");\n"
  "		}\n"
  "ENDVERBATIM\n"
  "		: the old standby. Cannot use if reproducible parallel sim\n"
  "		: independent of nhost or which host this instance is on\n"
  "		: is desired, since each instance on this cpu draws from\n"
  "		: the same stream\n"
  "		erand = exprand(1)\n"
  "VERBATIM\n"
  "	}\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "PROCEDURE noiseFromRandom() {\n"
  "VERBATIM\n"
  " {\n"
  "	void** pv = (void**)(&_p_donotuse);\n"
  "	if (ifarg(1)) {\n"
  "		*pv = nrn_random_arg(1);\n"
  "	}else{\n"
  "		*pv = (void*)0;\n"
  "	}\n"
  " }\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "PROCEDURE next_invl() {\n"
  "	if (number > 0) {\n"
  "		event = invl(interval)\n"
  "	}\n"
  "	if (ispike >= number) {\n"
  "		on = 0\n"
  "	}\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "	if (flag == 0) { : external event\n"
  "		if (w > 0 && on == 0) { : turn on spike sequence\n"
  "			: but not if a netsend is on the queue\n"
  "			init_sequence(t)\n"
  "			: randomize the first spike so on average it occurs at\n"
  "			: noise*interval (most likely interval is always 0)\n"
  "			next_invl()\n"
  "			event = event - interval*(1. - noise)\n"
  "			net_send(event, 1)\n"
  "		}else if (w < 0) { : turn off spiking definitively\n"
  "			on = 0\n"
  "		}\n"
  "	}\n"
  "	if (flag == 3) { : from INITIAL\n"
  "		if (on == 1) { : but ignore if turned off by external event\n"
  "			init_sequence(t)\n"
  "			net_send(0, 1)\n"
  "			net_send(2*check_interval,4)\n"
  "		}\n"
  "	}\n"
  "	if (flag == 1 && on == 1) {\n"
  "		ispike = ispike + 1\n"
  "		net_event(t)\n"
  "		next_invl()\n"
  "		transition = 0\n"
  "		if (on == 1) {\n"
  "			net_send(event, 1)\n"
  "		}\n"
  "	}	\n"
  "	if (flag == 5 && on == 1 && transition == 1) {         \n"
  "		ispike = ispike + 1\n"
  "        net_event(t)\n"
  "		next_invl()\n"
  "		if (on == 1) {\n"
  "			net_send(event, 5)\n"
  "		}\n"
  "    }	\n"
  "	if (flag == 4 && on == 1) { : check if interval has changed\n"
  "		if (interval < last_interval) { :if (2*interval < event || interval > 2*event) {\n"
  "			next_invl()\n"
  "			transition = 1\n"
  "			net_send(event,5)       \n"
  "		}\n"
  "		last_interval = interval\n"
  "		net_send(check_interval, 4) : next check interval event\n"
  "	}\n"
  "    if (flag == 1 && on == 0) { : For external PMd inputs - .event(timeStamp, 1) in server.py\n"
  "        net_event(t)\n"
  "    }	\n"
  "}\n"
  "\n"
  "COMMENT\n"
  "Presynaptic spike generator\n"
  "---------------------------\n"
  "\n"
  "This mechanism has been written to be able to use synapses in a single\n"
  "neuron receiving various types of presynaptic trains.  This is a \"fake\"\n"
  "presynaptic compartment containing a spike generator.  The trains\n"
  "of spikes can be either periodic or noisy (Poisson-distributed)\n"
  "\n"
  "Parameters;\n"
  "   noise: 	between 0 (no noise-periodic) and 1 (fully noisy)\n"
  "   interval: 	mean time between spikes (ms)\n"
  "   number: 	number of spikes (independent of noise)\n"
  "\n"
  "Written by Z. Mainen, modified by A. Destexhe, The Salk Institute\n"
  "\n"
  "Modified by Michael Hines for use with CVode\n"
  "The intrinsic bursting parameters have been removed since\n"
  "generators can stimulate other generators to create complicated bursting\n"
  "patterns with independent statistics (see below)\n"
  "\n"
  "Modified by Michael Hines to use logical event style with NET_RECEIVE\n"
  "This stimulator can also be triggered by an input event.\n"
  "If the stimulator is in the on==0 state (no net_send events on queue)\n"
  " and receives a positive weight\n"
  "event, then the stimulator changes to the on=1 state and goes through\n"
  "its entire spike sequence before changing to the on=0 state. During\n"
  "that time it ignores any positive weight events. If, in an on!=0 state,\n"
  "the stimulator receives a negative weight event, the stimulator will\n"
  "change to the on==0 state. In the on==0 state, it will ignore any ariving\n"
  "net_send events. A change to the on==1 state immediately fires the first spike of\n"
  "its sequence.\n"
  "\n"
  "ENDCOMMENT\n"
  ;
#endif
