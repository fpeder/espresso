#include "util.cuh"

static __global__
void ker_set2D(float *dst, const float v, int M, int N)
{
     int j=threadIdx.x + blockIdx.x * blockDim.x;
     int i=threadIdx.y + blockIdx.y * blockDim.y;

     if (i>=M || j>=N) return;

     dst[ID2(i,j,N)] = v;
}


static __global__
void ker_set3D(float *dst, const float v, int M, int N, int L)
{
     int k=threadIdx.x + blockIdx.x * blockDim.x;
     int j=threadIdx.y + blockIdx.y * blockDim.y;
     int i=threadIdx.z + blockIdx.z * blockDim.z;

     if (i>=M || j>=N || k>=L) return;

     dst[ID3(i,j,k,N,L)] = v;
}
