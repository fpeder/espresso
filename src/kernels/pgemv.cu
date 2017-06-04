#include "util.cuh"
#include "pgemv.cuh"

template <const int DIM_X, const int DIM_Y, const int TS>

static
void pgemv_template(const int m, const int n,
                    const uint64_t * __restrict__ A, int lda,
                    const uint64_t * __restrict__ x,
                    float *          __restrict__ y)
{
     dim3 grid(CEIL(m, TS), 1);
     dim3 threads(DIM_X, DIM_Y);

     pgemv_kernel <DIM_X, DIM_Y, TS>
          <<<grid, threads>>>
          (m, n, A, lda, x, y);

}


void pgemv(const int m, const int n,
           const uint64_t * __restrict__ A,
           const uint64_t * __restrict__ x,
           float *          __restrict__ y)
{
     pgemv_template <128, 1, 128>
          (m, n, A, n, x, y);
}
