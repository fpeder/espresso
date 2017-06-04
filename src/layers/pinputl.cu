#include "layers/cupinput.h"


// cupinputLayer cupinputLayer_init()
// {
//      cupinputLayer il; il.out.data=NULL;
//      return il;
// }

// void cupinputLayer_forward(ftens *t, cupinputLayer *il)
// {
//      int D=t->D, M=t->M, N=t->N, L=t->L; cuftens tmp;
//      if (!CUFMEM)
//           tmp = cuftens_convert(t);
//      else {
//           tmp = cuftens_from_cufmem(D, M, N, L);
//           cudaMemcpy(tmp.data, t->data, t->bytes, cuHtoD);
//      }

//      if (!il->out.data) il->out = cuptens_init(D, M, N, L);
//      cuptens_convert(&tmp, &il->out);

//      if(!d_fscratch) cuftens_free(&tmp);
// }

// void cupinputLayer_free(cupinputLayer *il)
// {
//      cuptens_free(&il->out);
// }
