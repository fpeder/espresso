#include "util.cuh"

template <typename T> static __global__
void ker_pad2D(const T * __restrict__ src,
               T       * __restrict__ dst,  int p,
               int Ms, int Ns, int Md, int Nd)
{
     int j=threadIdx.x + blockIdx.x*blockDim.x;
     int i=threadIdx.y + blockIdx.y*blockDim.y;

     if (i>=Ms || j>=Ns) return;

     dst[ID2(i+p,j+p,Nd)] = src[ID2(i,j,Ns)];
}


template <typename T> static __global__
void ker_pad3D(const T * __restrict__ src,
               T       * __restrict__ dst, int p,
               int Ms, int Ns, int Md, int Nd, int L)
{
     int k=threadIdx.x + blockIdx.x*blockDim.x;
     int j=threadIdx.y + blockIdx.y*blockDim.y;
     int i=threadIdx.z + blockIdx.z*blockDim.z;

     if (i>=Ms || j>=Ns || k>=L) return;

     dst[ID3(i+p,j+p,k,Nd,L)] = src[ID3(i,j,k,Ns,L)];
}


template <typename T>
void cupad_template(T *src, T *dst, int p, int D, int L,
                    int Ms, int Ns, int Md, int Nd)
{
     if (L == 1) {
          const int BS = 16;
          dim3 grid(CEIL(Ns, BS), CEIL(Ms, BS));
          dim3 block(BS, BS);
          for (int w=0; w < D; w++) {
               T *s = src + w * Ms*Ns*L;
               T *d = dst + w * Md*Nd*L;
               ker_pad2D <T> <<<grid, block>>>
                    (s, d, p, Ms, Ns, Md, Nd);
          }
     } else {
          const int BS = 8;
          dim3 grid(CEIL(L, BS), CEIL(Ns, BS), CEIL(Ms, BS));
          dim3 block(BS, BS, BS);
          for (int w=0; w < D; w++) {
               T *s = src + w * Ms*Ns*L;
               T *d = dst + w * Md*Nd*L;
               ker_pad3D <T> <<<grid, block>>>
                    (s, d, p, Ms, Ns, Md, Nd, L);
          }
     }
}
