#define fetch(A, m, n, bound) offs_##A[MIN((m)*LD##A+n, bound)]
#define cazA  (blx*BLK_M*LDA + idxA*LDA + idyA)
#define cazB  (bly*BLK_N*LDB + idxB*LDB + idyB)

template <int DIM_X,  int DIM_Y,  int BLK_M,  int BLK_N, int BLK_K,
          int DIM_XA, int DIM_YA, int DIM_XB, int DIM_YB,
          int THR_M,  int THR_N>

static __global__
void ker_sgemm(int M, int N, int K,
               const float * __restrict__ A, int LDA,
               const float * __restrict__ B, int LDB,
               float *       __restrict__ C, int LDC)
{
     int blx=blockIdx.y,  bly=blockIdx.x;
     int idx=threadIdx.y, idy=threadIdx.x,  idt=idx*DIM_Y+idy;
     int idxA=idt/DIM_YA, idyA=idt % DIM_YA;
     int idxB=idt/DIM_YB, idyB=idt % DIM_YB;

     float rC[THR_M][THR_N], rA[THR_M], rB[THR_N];
     float ra[BLK_M/DIM_XA][BLK_K/DIM_YA];
     float rb[BLK_N/DIM_XB][BLK_K/DIM_YB];

     __shared__ float sA[BLK_M][BLK_K+1];
     __shared__ float sB[BLK_N][BLK_K+1];

     const float *offs_A = A + cazA;
     const float *offs_B = B + cazB;
     ptrdiff_t boundA = (LDA*(M-1)+K) - cazA - 1;
     ptrdiff_t boundB = (LDB*(N-1)+K) - cazB - 1;

     int m, n, k, kk;
     #pragma unroll
     for (m=0; m<THR_M; m++)
          #pragma unroll
          for (n=0; n<THR_N; n++)
               rC[m][n] = 0.0f;

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
                         rC[m][n] += rA[m] * rB[n];
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

     kk = K - kk;
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
                    rC[m][n] += rA[m]*rB[n];
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

#undef cazA
#undef cazB
#undef fetch
