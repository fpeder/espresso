#include "cutens.h"
#include "util.cuh"
#include "sgemm.cuh"


template <int DIM_X,  int DIM_Y,  int BLK_M,  int BLK_N, int BLK_K,
          int DIM_XA, int DIM_YA, int DIM_XB, int DIM_YB>

static void
sgemm_template(int M, int N, int K,
               const float * __restrict__ A,  int LDA,
               const float * __restrict__ B,  int LDB,
               float *       __restrict__ C,  int LDC)
{
     dim3 dimBlock(DIM_Y, DIM_X);
     dim3 dimGrid(CEIL(N, BLK_N), CEIL(M, BLK_M));

     ker_sgemm  <DIM_X,  DIM_Y,  BLK_M,  BLK_N, BLK_K,
                 DIM_XA, DIM_YA, DIM_XB, DIM_YB,
                 BLK_M/DIM_X, BLK_N/DIM_Y>

          <<<dimGrid, dimBlock>>>
          (M, N, K, A, LDA, B, LDB, C, LDC);
}


void sgemm(int M, int N, int K,
           const float * __restrict__ A, int lda,
           const float * __restrict__ B, int ldb,
           float *       __restrict__ C, int ldc)
{
     sgemm_template <16,16, 96,96,16, 32,8, 32,8>
          (M, N, K, A, K, B, K, C, N);
}


void sgemm(cuftens *a, cuftens *b, cuftens *c)
{
     const int D=a->D, M=a->M, N=b->M, K=a->N;
     cuASSERT(a->N == b->N, "err:  shape\n");
     sgemm_template <16,16, 96,96,16, 32,8, 32,8>
          (D*M, N, K, a->data, K, b->data, K, c->data, N);
}
