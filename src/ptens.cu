#include "util.cuh"
#include "cuptens.h"
#include "kernels.h"


cuptens cuptens_empty(int D, int M, int N, int L)
{
     DIM_CHECK(D, M, N, L);
     cuptens out = {D, M, N, L};
     out.X       =  L>1 ? CEIL(L, 64) : CEIL(N, 64);
     out.p       = (L>1 ? L     : N) - out.X+1;
     out.MNL     = (L>1 ? M * N : M) * out.X;
     out.bytes   = BYTES(uint64_t, D * out.MNL);
     out.data    = NULL;
     return out;
}

cuptens cuptens_init(int D, int M, int N, int L)
{
     cuptens out = cuptens_empty(D, M, N, L);
     cudaMalloc(&out.data, out.bytes);
     cuASSERT(out.data, "err: cuptens malloc\n");
     //cudaMemset(&(out.data), 0, out.bytes);
     return out;
}


cuptens cuptens_from_cupmem(int D, int M, int N, int L)
{
     cuptens out = cuptens_empty(D, M, N, L);
     out.data = cupmem_reserve(out.bytes);
     return out;
}

cuptens cuptens_pad(cuptens *src, int p)
{
     cuASSERT(src->data && src->L > 1, "err: L>1 for pad\n");
     int D=src->D, M=src->M, N=src->N, L=src->L;
     cuptens out = cuptens_init(D, PAD(M,p), PAD(N,p), L);
     cupad(src, &out, p);
     return out;
}

cuptens cuptens_lower_init(cuptens *src, int W, int H, int Sx, int Sy)
{
     cuASSERT(src->data && src->L > 1, "err\n");
     int D=src->D, L=src->X;
     int Md = OUT_LEN(src->M, H, Sy);
     int Nd = OUT_LEN(src->N, W, Sx);
     return cuptens_init(D, Md, Nd, W*H*L*64);
}

cuptens cuptens_lower_cupmem(cuptens *src,
                             int W, int H, int Sx, int Sy)
{
     cuASSERT(src->data && src->L > 1, "err\n");
     int D=src->D, L=src->X;
     int Md = OUT_LEN(src->M, H, Sy);
     int Nd = OUT_LEN(src->N, W, Sx);
     cuptens out = cuptens_empty(D, Md, Nd, W*H*L*64);
     out.data = cupmem_reserve(out.bytes);
     return out;
}

void cuptens_free(cuptens *t)
{
     if (t->data) {cudaFree(t->data); t->data=NULL;}
}

cuptens cuptens_convert(ftens *t)
{
     cuASSERT(t->data, "err\n");
     cuftens tmp = cuftens_convert(t);
     cuptens out = cuptens_convert(&tmp);
     cuftens_free(&tmp);
     return out;
}

cuptens cuptens_convert(cuftens *t)
{
     cuftens tmp = cuftens_round_up(t, 64);
     cuptens out = cuptens_init(tmp.D, tmp.M, tmp.N, tmp.L);
     cupack(&tmp, &out);
     cuftens_free(&tmp);
     return out;
}

void cuptens_convert(cuftens *src, cuptens *dst)
{
     cupack(src, dst);
}

uint64_t *cuptens_dump(cuptens *t)
{
     uint64_t *out = MALLOC(uint64_t, t->bytes);
     cuASSERT(out, "err: dump malloc\n");
     cudaMemcpy(out, t->data, t->bytes, cuDtoH);
     return out;
}

// void cuptens_print_ch(cuptens *t, int w, int k, const char *fmt)
// {
//      int D=t->D, M=t->M, N=t->N, L=t->L, X=t->X;

//      if (!t->data) {printf("err: cuptens null\n"); return;}

//      uint64_t *a = cuptens_dump(t);
//      const uint64_t *b = a + w*t->MNL;

//      for (int i=0; i < M; i++) {
//           for (int j=0; j < N; j++) {
//                int v, p, o;
//                p=k>>6; o=k&63;
//                v = (b[ID3(i,j,p,N,X)] >> o) &1;
//                printf(fmt, 2*v -1);
//           }
//           NL;
//      }
//      free(a);
// }

void cuptens_print(cuptens *t)
{
     cuptens_print(t, "%2d");
}

void cuptens_print(cuptens *t, const char *fmt)
{
     int D=t->D, M=t->M, N=t->N, L=t->L, X=t->X;

     if (!t->data) {printf("err: cuptens null\n"); return;}

     uint64_t *a = cuptens_dump(t);

     for (int w=0; w < D; w++) {
          const uint64_t *b = a + w*t->MNL;
          for (int i=0; i < M; i++) {
               for (int k=0; k < L; k++) {
                    for (int j=0; j < N; j++) {
                         int v, p, o;
                         if (L == 1) {
                              p=j>>6; o=j&63;
                              v = (b[ID2(i,p,X)] >> o) &1;
                         } else {
                              p=k>>6; o=k&63;
                              v = (b[ID3(i,j,p,N,X)] >> o) &1;
                         }

                         printf(fmt, 2*v -1);

                    } SEP; } NL; } NL; }
     free(a);
}

void cuptens_print_shape(cuptens *t)
{
     printf("cuptens: %d %d %d %d %d\n", t->D, t->M, t->N, t->L, t->X);
}
