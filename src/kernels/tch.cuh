#include "util.cuh"

static __global__
void ker_tch (const float * __restrict__ src,
              float *       __restrict__ dst,
              const int M, const int N, const int L)
{
     int k=threadIdx.x + blockIdx.x * blockDim.x;
     int j=threadIdx.y + blockIdx.y * blockDim.y;
     int i=threadIdx.z + blockIdx.z * blockDim.z;

     if (i >= M || j >= N || k >= L) return;

     dst[ID3(j,k,i,L,M)] = src[ID3(i,j,k,N,L)];
}
