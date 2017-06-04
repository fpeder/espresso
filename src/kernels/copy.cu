#include "copy.cuh"
#include "cutens.h"


void cucopy(cuftens *src, cuftens *dst)
{
     int Ms=src->D*src->M, Ns=src->N, Ls=src->L;
     int Md=dst->D*dst->M, Nd=dst->N, Ld=dst->L;

     if (Ls > 1 && Ld > 1) {
          int TS = 8;
          dim3 grid(CEIL(Ls,TS), CEIL(Ns,TS), CEIL(Ms,TS));
          dim3 block(TS, TS, TS);

          ker_copy3D <<<grid, block>>>
               (src->data, dst->data, Ms, Ns, Ls, Md, Nd, Ld);
     } else {
          int TS = 16;
          dim3 grid(CEIL(Ns, TS), CEIL(Ms, TS));
          dim3 block(TS, TS);

          ker_copy2D <<<grid, block>>>
               (src->data, dst->data, Ms, Ns, Md, Nd);
     }
}
