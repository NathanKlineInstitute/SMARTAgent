#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _nsloc_reg(void);
extern void _stdp_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," nsloc.mod");
    fprintf(stderr," stdp.mod");
    fprintf(stderr, "\n");
  }
  _nsloc_reg();
  _stdp_reg();
}
