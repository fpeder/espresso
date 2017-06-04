#include "tch.cuh"
#include "cutens.h"


void cutch(cuftens *src, cuftens *dst)
{
     int M=src->M, N=src->N, L=src->L;
     cuASSERT(src->MNL == dst->MNL && src->D == dst->D,
              "err: cuth shape\n");

     int TS=8;
     dim3 blocks(CEIL(L, TS), CEIL(N, TS), CEIL(M, TS));
     dim3 threads(TS, TS, TS);

     for (int w = 0; w < src->D; w++) {
          float *s = src->data + w * src->MNL;
          float *d = dst->data + w * dst->MNL;
          ker_tch <<<blocks, threads>>>
               (s, d, M, N, L);
     }
}
