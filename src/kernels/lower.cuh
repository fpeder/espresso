#include "util.cuh"


#define SM_FILL(sm) {                                                   \
          sm[ID3(i,j,k,X,Z)] = src[ID3(I,J,K,Ns,L)];                    \
          if (i<H-1 && I+TM<Ms)                                         \
               sm[ID3(i+TM,j,k,X,Z)] = src[ID3(I+TM,J,K,Ns,L)];         \
          if (j<W-1 && J+TN<Ns)                                         \
               sm[ID3(i,j+TN,k,X,Z)] = src[ID3(I,J+TN,K,Ns,L)];         \
          if (i<H-1 && j<W-1 && I+TM<Ms && J+TN<Ns)                     \
               sm[ID3(i+TM,j+TN,k,X,Z)] = src[ID3(I+TM,J+TN,K,Ns,L)];   \
     }

#define DST_FILL(sm) {                                           \
          int l=0, R=(I/Sy*Nd + J/Sx)*Ld;                        \
          for (int y=0; y < H; y++)                              \
               for (int x=0; x < W; x++) {                       \
                    dst[R+l*L+K] = sm[ID3(i+y,j+x,k,X,Z)];       \
                    l++;                                         \
               }                                                 \
     }                                                           \


template <int TM, int TN, int TL> static __global__
void ker_lower(const float * __restrict__ src,
               float       * __restrict__ dst,
               int Ms, int Ns, int Md, int Nd, int L,
               int W,  int H,  int Sx, int Sy)
{
     int k=threadIdx.x, K=k+blockIdx.x * blockDim.x;
     int j=threadIdx.y, J=j+blockIdx.y * blockDim.y;
     int i=threadIdx.z, I=i+blockIdx.z * blockDim.z;
     const int X=TN+W-1, Z=TL, Ld=W*H*L;

     if (I>=Ms || J>=Ns || K>=L) return;

     extern __shared__ float sm[];

     SM_FILL(sm);

     // sm[ID3(i,j,k,X,Z)] = src[ID3(I,J,K,Ns,L)];
     // if (i<H-1 && I+TM<Ms)
     //      sm[ID3(i+TM,j,k,X,Z)] = src[ID3(I+TM,J,K,Ns,L)];

     // if (j<W-1 && J+TN<Ns)
     //      sm[ID3(i,j+TN,k,X,Z)] = src[ID3(I,J+TN,K,Ns,L)];

     // if (i<H-1 && j<W-1 && I+TM<Ms && J+TN<Ns)
     //      sm[ID3(i+TM,j+TN,k,X,Z)] = src[ID3(I+TM,J+TN,K,Ns,L)];

     __syncthreads();


     if (I/Sy>=Md || J/Sx>=Nd || K>=L) return;

     if ((I % Sy) == 0 && (J % Sx) == 0) {
          DST_FILL(sm);
          // int l=0, R=(I/Sy*Nd + J/Sx)*Ld;
          // for (int y=0; y < H; y++)
          //      for (int x=0; x < W; x++) {
          //           dst[R+l*L+K] = sm[ID3(i+y,j+x,k,X,Z)];
          //           l++;
          //      }
     }
}


template <int TM, int TN, int TL> static __global__
void ker_plower(const uint64_t * __restrict__ src,
                uint64_t       * __restrict__ dst,
                int Ms, int Ns, int Md, int Nd, int L,
                int W,  int H,  int Sx, int Sy)
{
     int k=threadIdx.x, K=k+blockIdx.x * blockDim.x;
     int j=threadIdx.y, J=j+blockIdx.y * blockDim.y;
     int i=threadIdx.z, I=i+blockIdx.z * blockDim.z;
     const int X=TN+W-1, Z=TL, Ld=W*H*L;

     if (I>=Ms || J>=Ns || K>=L) return;

     extern __shared__ uint64_t psm[];

     SM_FILL(psm);

     // psm[ID3(i,j,k,X,Z)] = src[ID3(I,J,K,Ns,L)];
     // if (i<H-1 && I+TM<Ms)
     //      psm[ID3(i+TM,j,k,X,Z)] = src[ID3(I+TM,J,K,Ns,L)];

     // if (j<W-1 && J+TN<Ns)
     //      psm[ID3(i,j+TN,k,X,Z)] = src[ID3(I,J+TN,K,Ns,L)];

     // if (i<H-1 && j<W-1 && I+TM<Ms && J+TN<Ns)
     //      psm[ID3(i+TM,j+TN,k,X,Z)] = src[ID3(I+TM,J+TN,K,Ns,L)];

     __syncthreads();


     if (I/Sy>=Md || J/Sx>=Nd || K>=L) return;

     if ((I % Sy) == 0 && (J % Sx) == 0) {
          DST_FILL(psm);
          // int l=0, R=(I/Sy*Nd + J/Sx)*Ld;
          // for (int y=0; y < H; y++)
          //      for (int x=0; x < W; x++) {
          //           dst[R+l*L+K] = psm[ID3(i+y,j+x,k,X,Z)];
          //           l++;
          //      }
     }
}

#undef SM_FILL
#undef DST_FILL
