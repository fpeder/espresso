#include "util.cuh"


template <int B> static __global__
void ker_bpsplit(const float *a, uint32_t *b, int Ns, int Nd)
{
     int w=threadIdx.x, W=w + blockIdx.x * blockDim.x;
     int j=threadIdx.y, J=j + blockIdx.y * blockDim.y;

     __shared__ uint32_t sm[32];
     __shared__ uint32_t c[B];

     if (J >= Ns) return;

     sm[j] = (uint32_t) a[W*Ns + J];

     __syncthreads();

     #pragma unroll
     for (int i = 0; i < B; i++) {
           c[i] = __ballot((sm[j] & (1 << i)));
     }

     __syncthreads();

     if (j == 0) {
          uint32_t *ptr = b + W*B*Nd;
          for (int i=0; i < B; i++) {
               ptr[i*Nd + blockIdx.y] = c[i];
          }
     }
}

template <int B> static __global__
void ker_bpmerge(const float *a, float *b, float *fix, float norm,
                 int N, int N2)
{
     int i=threadIdx.x, I=i + blockIdx.x*blockDim.x;
     int j=threadIdx.y, J=j + blockIdx.y*blockDim.y;

     __shared__ float sm[B];

     if (J >= N) return;

     sm[i] = a[I*N + J];
     __syncthreads();

     if (i == 0) {
          int id = blockIdx.x * N + J;
          float c = 0.0f;

          #pragma unroll
          for (int i=0; i < B; i++)
               c += sm[i] * (1<<i)/norm;

          b[id] = c - (fix ? fix[J % N2] : 0.0f);
     }
}


// template <int B> static __global__
// void ker_bpsplit(const float * __restrict__ a,
//                  uint32_t    * __restrict__ b,
//                  const int N)
// {
//      int j=threadIdx.x, J=j + blockIdx.x*blockDim.x;

//      __shared__ uint32_t p[32];
//      __shared__ uint32_t c[B];

//      p[j] = (uint32_t) a[J];

//      __syncthreads();

//      #pragma unroll
//      for (int i=0; i < B; i++)
//           c[i] = __ballot((p[j] & (1<<i)));

//      if (j == 0) {
//           #pragma unroll
//           for (int i=0; i < B; i++)
//                b[(i*N + (J%N) + (J/N*B*N))>>5] = c[i];
//      }
// }

// template <int B> static __global__
// void ker_bpmerge(int D, int M, int N,
//                  const float * __restrict__ src,
//                  float       * __restrict__ dst,
//                  const float * __restrict__ b,
//                  float norm)
// {
//      int i=threadIdx.x, I=i + blockIdx.x*blockDim.x;
//      int j=threadIdx.y, J=j + blockIdx.y*blockDim.y;

//      if (I>=D*M || J>=N) return;

//      __shared__ float sm[B][32];
//      sm[i][j] = src[ID2(I,J,N)];

//      __syncthreads();

//      if (i == 0) {
//           float c = 0.0f;
//           #pragma unroll
//           for(int k=0; k < B; k++)
//                c += sm[k][j] * (float)(1<<k)/norm;

//           dst[ID2(blockIdx.x, J, N)] = c - (b ? b[N] : 0.0f);
//      }
// }
