#include <float.h>
#include "util.cuh"

static __global__
void ker_maxpool (const float * __restrict__ src,
                  float *       __restrict__ dst,
                  const int Ms, const int Ns,
                  const int Md, const int Nd, const int L,
                  const int W,  const int H,
                  const int Sx, const int Sy)
{
     int k=threadIdx.x + blockIdx.x * blockDim.x;
     int j=threadIdx.y + blockIdx.y * blockDim.y;
     int i=threadIdx.z + blockIdx.z * blockDim.z;

     int I=i*Sy, J=j*Sx;

     if (i >= Md || j >= Nd || k >= L) return;

     float val, max=FLT_MIN;
     for (int y=0; y < H; y++)
          for (int x=0; x < W; x++) {
               val = src[ID3(I+y,J+x,k,Ns,L)];
               if (val > max) max = val;
          }

     dst[ID3(i,j,k,Nd,L)] = max;
}
