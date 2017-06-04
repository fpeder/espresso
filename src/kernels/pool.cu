#include "pool.cuh"
#include "cutens.h"


void cumaxpool(cuftens *src, cuftens *dst,
               int W, int H, int Sx, int Sy)
{

     int D=src->D,  L=src->L;
     int Ms=src->M, Ns=src->N;
     int Md=dst->M, Nd=dst->N;

     cuASSERT(L == dst->L && D == dst->D, "err: cupool shape\n");

     int TS = 16;
     dim3 grid(CEIL(L, TS), CEIL(Nd, W), CEIL(Md, H));
     dim3 block(TS, W, H);

     for (int w = 0; w < D; w++) {
          float *s = src->data + w * src->MNL;
          float *d = dst->data + w * dst->MNL;
          ker_maxpool <<<grid, block>>>
               (s, d, Ms, Ns, Md, Nd, L, W, H, Sx, Sy);
     }
}
