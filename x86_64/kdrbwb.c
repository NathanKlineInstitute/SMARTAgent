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
 
#define nrn_init _nrn_init__Kdrbwb
#define _nrn_initial _nrn_initial__Kdrbwb
#define nrn_cur _nrn_cur__Kdrbwb
#define _nrn_current _nrn_current__Kdrbwb
#define nrn_jacob _nrn_jacob__Kdrbwb
#define nrn_state _nrn_state__Kdrbwb
#define _net_receive _net_receive__Kdrbwb 
#define iassign iassign__Kdrbwb 
#define rates rates__Kdrbwb 
#define states states__Kdrbwb 
 
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
#define gkdr _p[0]
#define ek _p[1]
#define phin _p[2]
#define ninf _p[3]
#define taon _p[4]
#define n _p[5]
#define ik _p[6]
#define Dn _p[7]
#define v _p[8]
#define _g _p[9]
#define _ion_ik	*_ppvar[0]._pval
#define _ion_dikdv	*_ppvar[1]._pval
 
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
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_fun2(void);
 static void _hoc_fun1(void);
 static void _hoc_fun3(void);
 static void _hoc_iassign(void);
 static void _hoc_max(void);
 static void _hoc_min(void);
 static void _hoc_rates(void);
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

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_Kdrbwb", _hoc_setdata,
 "fun2_Kdrbwb", _hoc_fun2,
 "fun1_Kdrbwb", _hoc_fun1,
 "fun3_Kdrbwb", _hoc_fun3,
 "iassign_Kdrbwb", _hoc_iassign,
 "max_Kdrbwb", _hoc_max,
 "min_Kdrbwb", _hoc_min,
 "rates_Kdrbwb", _hoc_rates,
 0, 0
};
#define fun2 fun2_Kdrbwb
#define fun1 fun1_Kdrbwb
#define fun3 fun3_Kdrbwb
#define max max_Kdrbwb
#define min min_Kdrbwb
 extern double fun2( _threadargsprotocomma_ double , double , double , double );
 extern double fun1( _threadargsprotocomma_ double , double , double , double );
 extern double fun3( _threadargsprotocomma_ double , double , double , double );
 extern double max( _threadargsprotocomma_ double , double );
 extern double min( _threadargsprotocomma_ double , double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gkdr_Kdrbwb", "mS/cm2",
 "ek_Kdrbwb", "mV",
 "ninf_Kdrbwb", "1",
 "taon_Kdrbwb", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double n0 = 0;
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
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[2]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Kdrbwb",
 "gkdr_Kdrbwb",
 "ek_Kdrbwb",
 "phin_Kdrbwb",
 0,
 "ninf_Kdrbwb",
 "taon_Kdrbwb",
 0,
 "n_Kdrbwb",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 10, _prop);
 	/*initialize range parameters*/
 	gkdr = 9;
 	ek = -90;
 	phin = 5;
 	_prop->param = _p;
 	_prop->param_size = 10;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 	_ppvar[0]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[1]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _kdrbwb_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 10, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kdrbwb /Users/daviddonofrio/netpyne_workplace/SMARTAgent/x86_64/kdrbwb.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int iassign(_threadargsproto_);
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
static int  iassign ( _threadargsproto_ ) {
   ik = ( 1e-3 ) * gkdr * pow( n , 4.0 ) * ( v - ek ) ;
    return 0; }
 
static void _hoc_iassign(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 iassign ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dn = ( ninf - n ) / taon ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taon )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taon)))*(- ( ( ( ninf ) ) / taon ) / ( ( ( ( - 1.0 ) ) ) / taon ) - n) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   double _lan , _lbn , _lq10 ;
 _lq10 = phin ;
   _lan = fun3 ( _threadargscomma_ _lv , - 34.0 , - 0.01 , - 10.0 ) ;
   _lbn = fun1 ( _threadargscomma_ _lv , - 44.0 , 0.125 , - 80.0 ) ;
   ninf = _lan / ( _lan + _lbn ) ;
   taon = 1. / ( ( _lan + _lbn ) * _lq10 ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double fun1 ( _threadargsprotocomma_ double _lv , double _lV0 , double _lA , double _lB ) {
   double _lfun1;
 _lfun1 = _lA * exp ( ( _lv - _lV0 ) / _lB ) ;
   
return _lfun1;
 }
 
static void _hoc_fun1(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  fun1 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
}
 
double fun2 ( _threadargsprotocomma_ double _lv , double _lV0 , double _lA , double _lB ) {
   double _lfun2;
 _lfun2 = _lA / ( exp ( ( _lv - _lV0 ) / _lB ) + 1.0 ) ;
   
return _lfun2;
 }
 
static void _hoc_fun2(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  fun2 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
}
 
double fun3 ( _threadargsprotocomma_ double _lv , double _lV0 , double _lA , double _lB ) {
   double _lfun3;
 if ( fabs ( ( _lv - _lV0 ) / _lB ) < 1e-6 ) {
     _lfun3 = _lA * _lB / 1.0 * ( 1.0 - 0.5 * ( _lv - _lV0 ) / _lB ) ;
     }
   else {
     _lfun3 = _lA / 1.0 * ( _lv - _lV0 ) / ( exp ( ( _lv - _lV0 ) / _lB ) - 1.0 ) ;
     }
   
return _lfun3;
 }
 
static void _hoc_fun3(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  fun3 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
}
 
double min ( _threadargsprotocomma_ double _lx , double _ly ) {
   double _lmin;
 if ( _lx <= _ly ) {
     _lmin = _lx ;
     }
   else {
     _lmin = _ly ;
     }
   
return _lmin;
 }
 
static void _hoc_min(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  min ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double max ( _threadargsprotocomma_ double _lx , double _ly ) {
   double _lmax;
 if ( _lx >= _ly ) {
     _lmax = _lx ;
     }
   else {
     _lmax = _ly ;
     }
   
return _lmax;
 }
 
static void _hoc_max(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  max ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  n = n0;
 {
   rates ( _threadargscomma_ v ) ;
   n = ninf ;
   iassign ( _threadargs_ ) ;
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
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   iassign ( _threadargs_ ) ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
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
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(n) - _p;  _dlist1[0] = &(Dn) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/daviddonofrio/netpyne_workplace/SMARTAgent/mod/kdrbwb.mod";
static const char* nmodl_file_text = 
  "UNITS {\n"
  "  (mA) = (milliamp)\n"
  "  (mV) = (millivolt)\n"
  "  (mS) = (millisiemens)\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "  SUFFIX Kdrbwb\n"
  "  USEION k WRITE ik\n"
  "  RANGE phin,gkdr,ek\n"
  "  RANGE taon,ninf\n"
  "}\n"
  "	\n"
  "PARAMETER {\n"
  "  gkdr =   9 (mS/cm2)\n"
  "  ek   = -90 (mV)\n"
  "  phin = 5\n"
  "}\n"
  "    \n"
  "ASSIGNED {\n"
  "  v       (mV)\n"
  "  ik      (mA/cm2)\n"
  "  celsius (degC)\n"
  "  ninf    (1)\n"
  "  taon    (ms)\n"
  "}\n"
  "\n"
  "STATE { n }\n"
  "\n"
  "PROCEDURE iassign () { ik = (1e-3) * gkdr * n^4 * (v-ek) }\n"
  "\n"
  "INITIAL { \n"
  "  rates(v)\n"
  "  n  = ninf\n"
  "  iassign()\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "  SOLVE states METHOD cnexp	\n"
  "  iassign()\n"
  "}\n"
  "\n"
  "DERIVATIVE states { \n"
  "  rates(v)\n"
  "  n' = (ninf-n)/taon\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) { LOCAL an, bn, q10\n"
  "  q10  = phin:^((celsius-27.0(degC))/10.0(degC))\n"
  "    \n"
  "  an = fun3(v,  -34,  -0.01,   -10)\n"
  "  bn = fun1(v,  -44,   0.125,  -80)\n"
  "    \n"
  "  ninf = an/(an+bn)\n"
  "  taon = 1./((an+bn)*q10)\n"
  "}\n"
  "\n"
  ":::INCLUDE \"aux_fun.inc\"\n"
  ":::realpath /Users/daviddonofrio/netpyne_workplace/SMARTAgent/mod/aux_fun.inc\n"
  ": $Id: aux_fun.inc,v 1.1 2009/11/04 01:24:52 samn Exp $ \n"
  "COMMENT\n"
  "\n"
  "//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
  "//\n"
  "// NOTICE OF COPYRIGHT AND OWNERSHIP OF SOFTWARE\n"
  "//\n"
  "// Copyright 2007, The University Of Pennsylvania\n"
  "// 	School of Engineering & Applied Science.\n"
  "//   All rights reserved.\n"
  "//   For research use only; commercial use prohibited.\n"
  "//   Distribution without permission of Maciej T. Lazarewicz not permitted.\n"
  "//   mlazarew@seas.upenn.edu\n"
  "//\n"
  "//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "\n"
  ":-------------------------------------------------------------------\n"
  "FUNCTION fun1(v(mV),V0(mV),A(/ms),B(mV))(/ms) {\n"
  "\n"
  "	 fun1 = A*exp((v-V0)/B)\n"
  "}\n"
  "\n"
  "FUNCTION fun2(v(mV),V0(mV),A(/ms),B(mV))(/ms) {\n"
  "\n"
  "	 fun2 = A/(exp((v-V0)/B)+1)\n"
  "}\n"
  "\n"
  "FUNCTION fun3(v(mV),V0(mV),A(/ms),B(mV))(/ms) {\n"
  "\n"
  "    if(fabs((v-V0)/B)<1e-6) {\n"
  "    :if(v==V0) {\n"
  "        fun3 = A*B/1(mV) * (1- 0.5 * (v-V0)/B)\n"
  "    } else {\n"
  "        fun3 = A/1(mV)*(v-V0)/(exp((v-V0)/B)-1)\n"
  "    }\n"
  "}\n"
  "\n"
  "FUNCTION min(x,y) { if (x<=y){ min = x }else{ min = y } }\n"
  "FUNCTION max(x,y) { if (x>=y){ max = x }else{ max = y } }\n"
  ":::end INCLUDE aux_fun.inc\n"
  ;
#endif
