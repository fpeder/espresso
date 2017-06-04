#define fetch(A, m, n, bound) offs_##A[MIN((m)*LD##A+n, bound)]


__forceinline__ __device__
long long int int2_as_longlong (int2 a)
{
    long long int res;
    asm ("mov.b64 %0, {%1,%2};" : "=l"(res) : "r"(a.x), "r"(a.y));
    return res;
}


// __forceinline__ __device__
// void xcaz(int init, int *c, uint64_t a, uint64_t b)
// {
//      switch (init) {
//      case 0: *c += __popcll(a ^ b); break;
//      case 1: *c += __popcll(a & b) - __popcll((a ^ b) & b); break;
//      case 2: *c += __popcll(a & b) - __popcll((a ^ b) & a); break;
//      }
// }


template <const int DIM_X,  const int DIM_Y,
          const int BLK_M,  const int BLK_N, const int BLK_K,
          const int DIM_XA, const int DIM_YA,
          const int DIM_XB, const int DIM_YB,
          const int THR_M, const int THR_N>

static __global__
void pgemm_kernel(const int M, const int N, const int K,
                  const uint64_t * __restrict__ A, const int LDA,
                  const uint64_t * __restrict__ B, const int LDB,
                  float *       __restrict__ C, const int LDC,
                  int offsA, int offsB)
{
     int blx=blockIdx.y,  bly=blockIdx.x;
     int idx=threadIdx.y, idy=threadIdx.x,  idt=idx*DIM_Y+idy;
     int idxA=idt/DIM_YA, idyA=idt % DIM_YA;
     int idxB=idt/DIM_YB, idyB=idt % DIM_YB;

     int rC[THR_M][THR_N];
     uint64_t rA[THR_M], rB[THR_N];
     uint64_t ra[BLK_M/DIM_XA][BLK_K/DIM_YA];
     uint64_t rb[BLK_N/DIM_XB][BLK_K/DIM_YB];

     __shared__ uint64_t sA[BLK_M][BLK_K+1];
     __shared__ uint64_t sB[BLK_N][BLK_K+1];


#define cazA  (blx*BLK_M*LDA + idxA*LDA + idyA)
#define cazB  (bly*BLK_N*LDB + idxB*LDB + idyB)

#ifdef TEX1D
     int coord_A = offsA + cazA;
     int coord_B = offsB + cazB;
#else
     const uint64_t *offs_A = A + cazA;
     const uint64_t *offs_B = B + cazB;
     ptrdiff_t boundA = (LDA*(M-1)+K) - cazA - 1;
     ptrdiff_t boundB = (LDB*(N-1)+K) - cazB - 1;
#endif

#undef cazA
#undef cazB

     int m, n, k, kk;

     #pragma unroll
     for (m=0; m<THR_M; m++)
          #pragma unroll
          for (n=0; n<THR_N; n++)
               rC[m][n] = 0;

     #pragma unroll
     for (m=0; m<BLK_M; m+=DIM_XA)
          #pragma unroll
          for (n=0; n<BLK_K; n+=DIM_YA)
               sA[m+idxA][n+idyA] = fetch(A, m, n, boundA);

     #pragma unroll
     for (m=0; m<BLK_N; m+=DIM_XB)
          #pragma unroll
          for (n=0; n<BLK_K; n+=DIM_YB)
               sB[m+idxB][n+idyB] = fetch(B, m, n, boundB);

     __syncthreads();

     for (kk=0; kk<K-BLK_K; kk+=BLK_K) {
          #ifdef TEX1D
          coord_A += BLK_K;
          coord_B += BLK_K;
          #else
          offs_A += BLK_K; boundA -= BLK_K;
          offs_B += BLK_K; boundB -= BLK_K;
          #endif

          #pragma unroll
          for (m=0; m<BLK_M/DIM_XA; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YA; n++)
                    ra[m][n] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

          #pragma unroll
          for (m=0; m<BLK_N/DIM_XB; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YB; n++)
                    rb[m][n] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);


          #pragma unroll
          for (k=0; k<BLK_K; k++) {
               #pragma unroll
               for (m=0; m<THR_M; m++)
                    rA[m] = sA[m*DIM_X+idx][k];

               #pragma unroll
               for (n=0; n<THR_N; n++)
                    rB[n] = sB[n*DIM_Y+idy][k];

               #pragma unroll
               for (m=0; m<THR_M; m++)
                    #pragma unroll
                    for (n=0; n<THR_N; n++) {

                         rC[m][n] += __popcll(rA[m] ^ rB[n]);
                         //xcaz(INIT, &rC[m][n], rA[m], rB[n]);

                    }
          }

          __syncthreads();

          #pragma unroll
          for (m=0; m<BLK_M/DIM_XA; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YA; n++)
                    sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[m][n];

          #pragma unroll
          for (m=0; m<BLK_N/DIM_XB; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YB; n++)
                    sB[m*DIM_XB+idxB][n*DIM_YB+idyB] = rb[m][n];

          __syncthreads();

     }

     kk=K-kk;
     #pragma unroll
     for (k=0; k<kk; k++) {
          #pragma unroll
          for (m=0; m<THR_M; m++)
               rA[m] = sA[m*DIM_X+idx][k];

          #pragma unroll
          for (n=0; n<THR_N; n++)
               rB[n] = sB[n*DIM_Y+idy][k];

          #pragma unroll
          for (m=0; m<THR_M; m++)
               #pragma unroll
               for (n=0; n<THR_N;n++) {
                    rC[m][n] += __popcll(rA[m] ^ rB[n]);
                    //xcaz(INIT, &rC[m][n], rA[m], rB[n]);
               }
     }

     #pragma unroll
     for (m=0; m<THR_M; m++) {
          int i = blx*BLK_M + m*DIM_X + idx;
          #pragma unroll
          for (n=0; n<THR_N; n++) {
               int j = bly*BLK_N + n*DIM_Y + idy;
               if (i<M && j<N)
                    //if (INIT)
                    ///C[i*LDC+j] = rC[m][m];
                    //else
                    C[i*LDC+j] = (LDA<<6)-(rC[m][n]<<1);
          }
     }
}

//////////////////////////////////////////////////////////
template <const int DIM_X,  const int DIM_Y,
          const int BLK_M,  const int BLK_N, const int BLK_K,
          const int DIM_XA, const int DIM_YA,
          const int DIM_XB, const int DIM_YB,
          const int THR_M,  const int THR_N>

static __global__
void pgemm_kernel_init(const int M, const int N, const int K,
                       const uint64_t * __restrict__ A, const int LDA,
                       const uint64_t * __restrict__ B, const int LDB,
                       float *       __restrict__ C, const int LDC,
                       int offsA, int offsB)
{
     int blx=blockIdx.y,  bly=blockIdx.x;
     int idx=threadIdx.y, idy=threadIdx.x,  idt=idx*DIM_Y+idy;
     int idxA=idt/DIM_YA, idyA=idt % DIM_YA;
     int idxB=idt/DIM_YB, idyB=idt % DIM_YB;

     int rC[THR_M][THR_N];
     uint64_t rA[THR_M], rB[THR_N];
     uint64_t ra[BLK_M/DIM_XA][BLK_K/DIM_YA];
     uint64_t rb[BLK_N/DIM_XB][BLK_K/DIM_YB];

     __shared__ uint64_t sA[BLK_M][BLK_K+1];
     __shared__ uint64_t sB[BLK_N][BLK_K+1];


#define cazA  (blx*BLK_M*LDA + idxA*LDA + idyA)
#define cazB  (bly*BLK_N*LDB + idxB*LDB + idyB)

     const uint64_t *offs_A = A + cazA;
     const uint64_t *offs_B = B + cazB;
     ptrdiff_t boundA = (LDA*(M-1)+K) - cazA - 1;
     ptrdiff_t boundB = (LDB*(N-1)+K) - cazB - 1;

#undef cazA
#undef cazB

     int m, n, k, kk;

     #pragma unroll
     for (m=0; m<THR_M; m++)
          #pragma unroll
          for (n=0; n<THR_N; n++)
               rC[m][n] = 0;

     #pragma unroll
     for (m=0; m<BLK_M; m+=DIM_XA)
          #pragma unroll
          for (n=0; n<BLK_K; n+=DIM_YA)
               sA[m+idxA][n+idyA] = fetch(A, m, n, boundA);

     #pragma unroll
     for (m=0; m<BLK_N; m+=DIM_XB)
          #pragma unroll
          for (n=0; n<BLK_K; n+=DIM_YB)
               sB[m+idxB][n+idyB] = fetch(B, m, n, boundB);

     __syncthreads();

     for (kk=0; kk<K-BLK_K; kk+=BLK_K) {
          #ifdef TEX1D
          coord_A += BLK_K;
          coord_B += BLK_K;
          #else
          offs_A += BLK_K; boundA -= BLK_K;
          offs_B += BLK_K; boundB -= BLK_K;
          #endif

          #pragma unroll
          for (m=0; m<BLK_M/DIM_XA; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YA; n++)
                    ra[m][n] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

          #pragma unroll
          for (m=0; m<BLK_N/DIM_XB; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YB; n++)
                    rb[m][n] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);


          #pragma unroll
          for (k=0; k<BLK_K; k++) {
               #pragma unroll
               for (m=0; m<THR_M; m++)
                    rA[m] = sA[m*DIM_X+idx][k];

               #pragma unroll
               for (n=0; n<THR_N; n++)
                    rB[n] = sB[n*DIM_Y+idy][k];

               #pragma unroll
               for (m=0; m<THR_M; m++)
                    #pragma unroll
                    for (n=0; n<THR_N; n++) {
                         rC[m][n] += (__popcll(rA[m] & rB[n]) -
                                     __popcll((rA[m] ^ rB[n]) & rB[n]));
                    }
          }

          __syncthreads();

          #pragma unroll
          for (m=0; m<BLK_M/DIM_XA; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YA; n++)
                    sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[m][n];

          #pragma unroll
          for (m=0; m<BLK_N/DIM_XB; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YB; n++)
                    sB[m*DIM_XB+idxB][n*DIM_YB+idyB] = rb[m][n];

          __syncthreads();

     }

     kk=K-kk;
     #pragma unroll
     for (k=0; k<kk; k++) {
          #pragma unroll
          for (m=0; m<THR_M; m++)
               rA[m] = sA[m*DIM_X+idx][k];

          #pragma unroll
          for (n=0; n<THR_N; n++)
               rB[n] = sB[n*DIM_Y+idy][k];

          #pragma unroll
          for (m=0; m<THR_M; m++)
               #pragma unroll
               for (n=0; n<THR_N;n++) {
                    rC[m][n] += (__popcll(rA[m] & rB[n]) -
                                __popcll((rA[m] ^ rB[n]) & rB[n]));
               }
     }

     #pragma unroll
     for (m=0; m<THR_M; m++) {
          int i = blx*BLK_M + m*DIM_X + idx;
          #pragma unroll
          for (n=0; n<THR_N; n++) {
               int j = bly*BLK_N + n*DIM_Y + idy;
               if (i<M && j<N)
                    C[i*LDC+j] = rC[m][n];
          }
     }
}
///////////////////////////////////////////////////////////////
template <const int DIM_X,  const int DIM_Y,
          const int BLK_M,  const int BLK_N, const int BLK_K,
          const int DIM_XA, const int DIM_YA,
          const int DIM_XB, const int DIM_YB,
          const int THR_M,  const int THR_N>

static __global__
void pgemm_kernel_init_rev(const int M, const int N, const int K,
                           const uint64_t * __restrict__ A, const int LDA,
                           const uint64_t * __restrict__ B, const int LDB,
                           float *       __restrict__ C, const int LDC,
                           int offsA, int offsB)
{
     int blx=blockIdx.y,  bly=blockIdx.x;
     int idx=threadIdx.y, idy=threadIdx.x,  idt=idx*DIM_Y+idy;
     int idxA=idt/DIM_YA, idyA=idt % DIM_YA;
     int idxB=idt/DIM_YB, idyB=idt % DIM_YB;

     int rC[THR_M][THR_N];
     uint64_t rA[THR_M], rB[THR_N];
     uint64_t ra[BLK_M/DIM_XA][BLK_K/DIM_YA];
     uint64_t rb[BLK_N/DIM_XB][BLK_K/DIM_YB];

     __shared__ uint64_t sA[BLK_M][BLK_K+1];
     __shared__ uint64_t sB[BLK_N][BLK_K+1];


#define cazA  (blx*BLK_M*LDA + idxA*LDA + idyA)
#define cazB  (bly*BLK_N*LDB + idxB*LDB + idyB)

     const uint64_t *offs_A = A + cazA;
     const uint64_t *offs_B = B + cazB;
     ptrdiff_t boundA = (LDA*(M-1)+K) - cazA - 1;
     ptrdiff_t boundB = (LDB*(N-1)+K) - cazB - 1;

#undef cazA
#undef cazB

     int m, n, k, kk;

     #pragma unroll
     for (m=0; m<THR_M; m++)
          #pragma unroll
          for (n=0; n<THR_N; n++)
               rC[m][n] = 0;

     #pragma unroll
     for (m=0; m<BLK_M; m+=DIM_XA)
          #pragma unroll
          for (n=0; n<BLK_K; n+=DIM_YA)
               sA[m+idxA][n+idyA] = fetch(A, m, n, boundA);

     #pragma unroll
     for (m=0; m<BLK_N; m+=DIM_XB)
          #pragma unroll
          for (n=0; n<BLK_K; n+=DIM_YB)
               sB[m+idxB][n+idyB] = fetch(B, m, n, boundB);

     __syncthreads();

     for (kk=0; kk<K-BLK_K; kk+=BLK_K) {
          offs_A += BLK_K; boundA -= BLK_K;
          offs_B += BLK_K; boundB -= BLK_K;

          #pragma unroll
          for (m=0; m<BLK_M/DIM_XA; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YA; n++)
                    ra[m][n] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

          #pragma unroll
          for (m=0; m<BLK_N/DIM_XB; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YB; n++)
                    rb[m][n] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);


          #pragma unroll
          for (k=0; k<BLK_K; k++) {
               #pragma unroll
               for (m=0; m<THR_M; m++)
                    rA[m] = sA[m*DIM_X+idx][k];

               #pragma unroll
               for (n=0; n<THR_N; n++)
                    rB[n] = sB[n*DIM_Y+idy][k];

               #pragma unroll
               for (m=0; m<THR_M; m++)
                    #pragma unroll
                    for (n=0; n<THR_N; n++)
                         rC[m][n] += __popcll(rA[m] & rB[n]) -
                              __popcll((rA[m] ^ rB[n]) & rA[n]);
          }

          __syncthreads();

          #pragma unroll
          for (m=0; m<BLK_M/DIM_XA; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YA; n++)
                    sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[m][n];

          #pragma unroll
          for (m=0; m<BLK_N/DIM_XB; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YB; n++)
                    sB[m*DIM_XB+idxB][n*DIM_YB+idyB] = rb[m][n];

          __syncthreads();

     }

     kk=K-kk;
     #pragma unroll
     for (k=0; k<kk; k++) {
          #pragma unroll
          for (m=0; m<THR_M; m++)
               rA[m] = sA[m*DIM_X+idx][k];

          #pragma unroll
          for (n=0; n<THR_N; n++)
               rB[n] = sB[n*DIM_Y+idy][k];

          #pragma unroll
          for (m=0; m<THR_M; m++)
               #pragma unroll
               for (n=0; n<THR_N;n++)
                    rC[m][n] += __popcll(rA[m] & rB[n]) -
                         __popcll((rA[m] ^ rB[n]) & rA[n]);
     }

     #pragma unroll
     for (m=0; m<THR_M; m++) {
          int i = blx*BLK_M + m*DIM_X + idx;
          #pragma unroll
          for (n=0; n<THR_N; n++) {
               int j = bly*BLK_N + n*DIM_Y + idy;
               if (i<M && j<N)
                    C[i*LDC+j] = rC[m][n];
          }
     }
}
///////////////////////////////////////////////////////////////
template <const int INIT,
          const int DIM_X,  const int DIM_Y,
          const int BLK_M,  const int BLK_N, const int BLK_K,
          const int DIM_XA, const int DIM_YA,
          const int DIM_XB, const int DIM_YB,
          const int THR_M, const int THR_N>

static __global__
void pgemm32_kernel(const int M, const int N, const int K,
                    const uint32_t * __restrict__ A, const int LDA,
                    const uint32_t * __restrict__ B, const int LDB,
                    float *       __restrict__ C, const int LDC,
                    int offsA, int offsB)
{
     int blx=blockIdx.y,  bly=blockIdx.x;
     int idx=threadIdx.y, idy=threadIdx.x,  idt=idx*DIM_Y+idy;
     int idxA=idt/DIM_YA, idyA=idt % DIM_YA;
     int idxB=idt/DIM_YB, idyB=idt % DIM_YB;

     int rC[THR_M][THR_N];
     uint32_t rA[THR_M], rB[THR_N];
     uint32_t ra[BLK_M/DIM_XA][BLK_K/DIM_YA];
     uint32_t rb[BLK_N/DIM_XB][BLK_K/DIM_YB];

     __shared__ uint32_t sA[BLK_M][BLK_K+1];
     __shared__ uint32_t sB[BLK_N][BLK_K+1];

#define cazA  (blx*BLK_M*LDA + idxA*LDA + idyA)
#define cazB  (bly*BLK_N*LDB + idxB*LDB + idyB)

     const uint32_t *offs_A = A + cazA;
     const uint32_t *offs_B = B + cazB;
     ptrdiff_t boundA = (LDA*(M-1)+K) - cazA - 1;
     ptrdiff_t boundB = (LDB*(N-1)+K) - cazB - 1;

#undef cazA
#undef cazB

     int m, n, k, kk;

     #pragma unroll
     for (m=0; m<THR_M; m++)
          #pragma unroll
          for (n=0; n<THR_N; n++)
               rC[m][n] = 0;

     #pragma unroll
     for (m=0; m<BLK_M; m+=DIM_XA)
          #pragma unroll
          for (n=0; n<BLK_K; n+=DIM_YA)
               sA[m+idxA][n+idyA] = fetch(A, m, n, boundA);

     #pragma unroll
     for (m=0; m<BLK_N; m+=DIM_XB)
          #pragma unroll
          for (n=0; n<BLK_K; n+=DIM_YB)
               sB[m+idxB][n+idyB] = fetch(B, m, n, boundB);

     __syncthreads();

     for (kk=0; kk<K-BLK_K; kk+=BLK_K) {
          offs_A += BLK_K; boundA -= BLK_K;
          offs_B += BLK_K; boundB -= BLK_K;

          #pragma unroll
          for (m=0; m<BLK_M/DIM_XA; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YA; n++)
                    ra[m][n] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

          #pragma unroll
          for (m=0; m<BLK_N/DIM_XB; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YB; n++)
                    rb[m][n] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);


          #pragma unroll
          for (k=0; k<BLK_K; k++) {
               #pragma unroll
               for (m=0; m<THR_M; m++)
                    rA[m] = sA[m*DIM_X+idx][k];

               #pragma unroll
               for (n=0; n<THR_N; n++)
                    rB[n] = sB[n*DIM_Y+idy][k];

               #pragma unroll
               for (m=0; m<THR_M; m++)
                    #pragma unroll
                    for (n=0; n<THR_N; n++)
                         rC[m][n] += __popc(rA[m] ^ rB[n]);

          }

          __syncthreads();

          #pragma unroll
          for (m=0; m<BLK_M/DIM_XA; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YA; n++)
                    sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[m][n];

          #pragma unroll
          for (m=0; m<BLK_N/DIM_XB; m++)
               #pragma unroll
               for (n=0; n<BLK_K/DIM_YB; n++)
                    sB[m*DIM_XB+idxB][n*DIM_YB+idyB] = rb[m][n];

          __syncthreads();

     }

     kk=K-kk;
     #pragma unroll
     for (k=0; k<kk; k++) {
          #pragma unroll
          for (m=0; m<THR_M; m++)
               rA[m] = sA[m*DIM_X+idx][k];

          #pragma unroll
          for (n=0; n<THR_N; n++)
               rB[n] = sB[n*DIM_Y+idy][k];

          #pragma unroll
          for (m=0; m<THR_M; m++)
               #pragma unroll
               for (n=0; n<THR_N;n++)
                    rC[m][n] += __popc(rA[m] ^ rB[n]);
     }

     #pragma unroll
     for (m=0; m<THR_M; m++) {
          int i = blx*BLK_M + m*DIM_X + idx;
          #pragma unroll
          for (n=0; n<THR_N; n++) {
               int j = bly*BLK_N + n*DIM_Y + idy;
               if (i<M && j<N)
                    C[i*LDC+j] = (LDA<<5)-(rC[m][n]<<1);
          }
     }
}
