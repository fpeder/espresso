#include "util.cuh"

__global__ static
void ker_copy2D (const float *src, float *dst,
                 const int Ms, const int Ns,
                 const int Md, const int Nd)
{
     int j=threadIdx.x + blockIdx.x*blockDim.x;
     int i=threadIdx.y + blockIdx.y*blockDim.y;

     if (i>=Ms || j>=Ns) return;

     dst[ID2(i,j,Nd)] = src[ID2(i,j,Ns)];
}


__global__ static
void ker_copy3D (const float *src, float *dst,
                 const int Ms, const int Ns, const int Ls,
                 const int Md, const int Nd, const int Ld)
{
     int k=threadIdx.x + blockIdx.x * blockDim.x;
     int j=threadIdx.y + blockIdx.y * blockDim.y;
     int i=threadIdx.z + blockIdx.z * blockDim.z;

     if (i>=Ms || j>=Ns || k>=Ls) return;

     dst[ID3(i,j,k,Nd,Ld)] = src[ID3(i,j,k,Ns,Ls)];
}
