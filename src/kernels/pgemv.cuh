template <const int DIM_X, const int DIM_Y, const int TS>

__global__
void pgemv_kernel(const int m, const int n,
                  const uint64_t * __restrict__ A, int lda,
                  const uint64_t * __restrict__ x,
                  float          * __restrict__ y)
{
     if (m <= 0 || n <= 0) return;

     int nt = blockDim.x * blockDim.y * blockDim.z;

     if (DIM_X * DIM_Y != nt) return;

     int tid = threadIdx.x + threadIdx.y * blockDim.x;
     int tx  = tid % DIM_X, ty = tid / DIM_X;
     int ind = blockIdx.x * TS + tx;

     __shared__ int sdata[DIM_X * DIM_Y];

     int st = blockIdx.x * TS;
     int ed = MIN(st + TS, ROUND_UP(m, DIM_X));
     int iters = (ed - st)/DIM_X;

     for (int i=0; i < iters; i++) {
          if (ind < m ) A += ind*lda;
          int res = 0;
          if (ind < m ) {
               for (int col=ty; col < n; col += DIM_Y)
                    res += __popcll(A[col] ^ x[col]);
          }

          sdata[ty + tx * DIM_Y] = res;

          __syncthreads();

          if (ty == 0 && ind < m) {
               for (int i=1; i < DIM_Y; i++)
                    sdata[tx * DIM_Y] += sdata[i + tx * DIM_Y];
          }

          if (ty == 0 && ind < m)
               y[ind] = (lda<<6) - (sdata[tx * DIM_Y]<<1);

          __syncthreads();

          if (ind < m) A -= ind*lda;
          ind += DIM_X;
     }
}
