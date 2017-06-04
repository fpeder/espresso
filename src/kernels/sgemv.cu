#include "sgemv.cuh"
#include "util.cuh"
#include "cutens.h"

template <const int DIM_X, const int DIM_Y, const int TS>

void sgemv_template(const int m, const int n,
                    const float * __restrict__ A, int lda,
                    const float * __restrict__ x,
                    float *       __restrict__ y)
{
     dim3 grid(CEIL(m, TS), 1);
     dim3 block(DIM_X, DIM_Y);

     ker_sgemv <DIM_X, DIM_Y, TS>
          <<<grid, block>>>
          (m, n, A, lda, x, y);
}


void sgemv (cuftens *a, cuftens *b, cuftens *c)
{
     const int M=a->M, N=a->N;
     cuASSERT(b->M==1 && b->N==N, "err: sgemv shape\n");
     sgemv_template <256, 1, 256>
          (M, N, a->data, N, b->data, c->data);
}
