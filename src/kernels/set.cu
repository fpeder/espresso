#include "set.cuh"
#include "cutens.h"


// void cuset2D(cuftens *t, const float v)
// {
//      const int BS=32;
//      const int M=t->M, N=t->N;

//      dim3 grid(CEIL(N, BS), CEIL(M, BS));
//      dim3 block(BS, BS);

//      ker_set2D <<<grid, block>>> (t->data, v, M, N);
// }


void cuset(cuftens *t, const float v)
{
     const int M=t->D*t->M, N=t->N, L=t->L;

     if (L > 1) {
          const int BS = 8;
          dim3 grid(CEIL(L, BS), CEIL(N, BS), CEIL(M, BS));
          dim3 block(BS, BS, BS);

          ker_set3D <<<grid, block>>>
               (t->data, v, M, N, L);

     } else {
          const int BS = 16;
          dim3 grid(CEIL(N, BS), CEIL(M, BS));
          dim3 block(BS, BS);

          ker_set2D <<<grid, block>>>
               (t->data, v, M, N);

     }
}
