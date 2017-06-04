#include "util.cuh"
#include "pgemm.cuh"


template <int INIT,
          int DIM_X,  int DIM_Y,  int BLK_M,  int BLK_N, int BLK_K,
          int DIM_XA, int DIM_YA, int DIM_XB, int DIM_YB>

static void
pgemm_template(int M, int N, int K,
               const uint64_t * __restrict__ A, int LDA,
               const uint64_t * __restrict__ B, int LDB,
               float *          __restrict__ C, int LDC)
{
     size_t offsA=0, offsB=0;
     offsA /= sizeof(A[0]);
     offsB /= sizeof(B[0]);

     dim3 dimBlock(DIM_Y, DIM_X);
     dim3 dimGrid(CEIL(N, BLK_N), CEIL(M, BLK_M));

     switch (INIT) {
     case 0:
          pgemm_kernel <DIM_X,  DIM_Y,  BLK_M,  BLK_N, BLK_K,
                        DIM_XA, DIM_YA, DIM_XB, DIM_YB,
                        BLK_M/DIM_X, BLK_N/DIM_Y>
               <<<dimGrid, dimBlock>>>
               (M, N, K, A, LDA, B, LDB, C, LDC, offsA, offsB);
          break;

     case 1:
          pgemm_kernel_init <DIM_X,  DIM_Y,  BLK_M,  BLK_N, BLK_K,
                             DIM_XA, DIM_YA, DIM_XB, DIM_YB,
                             BLK_M/DIM_X, BLK_N/DIM_Y>
               <<<dimGrid, dimBlock>>>
               (M, N, K, A, LDA, B, LDB, C, LDC, offsA, offsB);
          break;

     case 2:
          pgemm_kernel_init_rev <DIM_X,  DIM_Y,  BLK_M,  BLK_N, BLK_K,
                                 DIM_XA, DIM_YA, DIM_XB, DIM_YB,
                                 BLK_M/DIM_X, BLK_N/DIM_Y>
               <<<dimGrid, dimBlock>>>
               (M, N, K, A, LDA, B, LDB, C, LDC, offsA, offsB);
          break;
     }
}


void pgemm(int M, int N, int K,
           const uint64_t * __restrict__ A,
           const uint64_t * __restrict__ B,
           float *          __restrict__ C)
{
     pgemm_template
          <0, 16,16, 16,16,16, 16,16, 16,16>
          (M, N, K, A, K, B, K, C, N);
}


void pgemm_init(int M, int N, int K,
                const uint64_t * __restrict__ A,
                const uint64_t * __restrict__ B,
                float *          __restrict__ C)
{
     pgemm_template
          <1, 16,16, 16,16,16, 16,16, 16,16>
          (M, N, K, A, K, B, K, C, N);
}


void pgemm_init(int M, int N, int K,
                const uint64_t * __restrict__ a, int lda,
                const uint64_t * __restrict__ b, int ldb,
                float *          __restrict__ c, int ldc)
{
     pgemm_template <1, 16, 16, 16,16,16, 16,16, 16,16>
          (M, N, K, a, lda, b, ldb, c, ldc);
}


void pgemm_init_rev(const int M, const int N, const int K,
                    const uint64_t * __restrict__ A,
                    const uint64_t * __restrict__ B,
                    float *          __restrict__ C)
{
     pgemm_template
          <2, 16,16, 16,16,16, 16,16, 16,16>
          (M, N, K, A, K, B, K, C, N);
}

///////////////////////////////////////////////////////
template <int INIT,
          int DIM_X,  int DIM_Y,  int BLK_M,  int BLK_N, int BLK_K,
          int DIM_XA, int DIM_YA, int DIM_XB, int DIM_YB>

static void
pgemm32_template(int M, int N, int K,
                 const uint32_t * __restrict__ A, int LDA,
                 const uint32_t * __restrict__ B, int LDB,
                 float *          __restrict__ C, int LDC)
{
     size_t offsA=0, offsB=0;
     offsA /= sizeof(A[0]);
     offsB /= sizeof(B[0]);

     dim3 dimBlock(DIM_Y, DIM_X);
     dim3 dimGrid(CEIL(N, BLK_N), CEIL(M, BLK_M));

     pgemm32_kernel <INIT, DIM_X,  DIM_Y,  BLK_M,  BLK_N, BLK_K,
                     DIM_XA, DIM_YA, DIM_XB, DIM_YB,
                     BLK_M/DIM_X, BLK_N/DIM_Y>

          <<<dimGrid, dimBlock>>>
          (M, N, K, A, LDA, B, LDB, C, LDC, offsA, offsB);
}


void pgemm32(int M, int N, int K,
             const uint32_t * __restrict__ A,
             const uint32_t * __restrict__ B,
             float *          __restrict__ C)
{
     pgemm32_template
          <0, 16,16, 16,16,16, 16,16, 16,16>
          (M, N, K, A, K, B, K, C, N);
}
