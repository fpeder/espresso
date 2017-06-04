#include <string.h>
#include <float.h>

#include "tens.h"
#include "util.h"


ftens ftens_init(int D, int M, int N, int L)
{
     ftens t = {D, M, N, L, M*N*L, BYTES(float, D*M*N*L)};
     t.data = MALLOC(float, D*M*N*L);
     ASSERT(t.data, "err: ftens malloc");
     return t;
}

ftens ftens_from_ptr(int D, int M, int N, int L, float *ptr)
{
     ftens t = {D, M, N, L, M*N*L, BYTES(float, D*M*N*L)};
     ASSERT(ptr, "err: NULL ptr\n");
     t.data = ptr;
     return t;
}

void ftens_print_shape(ftens *t)
{
     printf("ftens: %d %d %d %d\n", t->D, t->M, t->N, t->L);
}

void ftens_free(ftens *t)
{
     if (t->data) {free(t->data); t->data=NULL;}
}

ftens ftens_copy(ftens *t)
{
     const int D=t->D, M=t->M, N=t->N, L=t->L;
     ftens out = ftens_init(D, M, N, L);
     ASSERT(t->data, "err: null tens\n");
     memcpy(out.data, t->data, t->bytes);
     return out;
}

ftens ftens_from_file(int D, int M, int N, int L, FILE *pf)
{
     ftens out = ftens_init(D, M, N, L);
     fread(out.data, sizeof(float), D*M*N*L, pf);
     return out;
}

void ftens_reshape(ftens *t, int D, int M, int N, int L)
{
     const int len = ftens_len(t);
     ASSERT(len== D*M*N*L, "err: ftens reshape\n");
     t->D=D; t->M=M; t->N=N; t->L=L;
}


void ftens_clear(ftens *t) {memset(t->data, 0, t->bytes);}

ftens ftens_zeros(int D, int M, int N, int L)
{
     ftens t = ftens_init(D, M, N, L);
     memset(t.data, 0, t.bytes);
     return t;
}

ftens ftens_ones(int D, int M, int N, int L)
{
     ftens t = ftens_init(D, M, N, L);
     for (int i=0; i < LEN(t); i++)
          D(t)[i] = 1.0f;
     return t;
}

ftens ftens_rand(int D, int M, int N, int L)
{
     ftens t = ftens_init(D, M, N, L);
     for (int i=0; i < LEN(t); i++)
          D(t)[i] = (rand() % 255) - 128.0f;
     return t;
}


void ftens_sign(ftens *t)
{
     for (int i=0; i < t->bytes/sizeof(float); i++)
          t->data[i] = 2.0f * (t->data[i] > 0.0f) - 1.0f;
}

ftens ftens_rand_range(int D, int M, int N, int L,
                       float min, float max)
{
     ftens t = ftens_init(D, M, N, L);
     for (int i=0; i < ftens_len(&t); i++)
          t.data[i] = ((max-min)*rand())/RAND_MAX + min;
     return t;
}

ftens ftens_copy_tch(ftens *a)
{
     const int M=a->M, N=a->N, L=a->L, D=a->D;
     ftens b = ftens_init(D, N, L, M);
     for (int w=0; w<D; w++) {
          float *src = a->data + w*a->MNL;
          float *dst = b. data + w*b. MNL;
          for (int i=0; i<M; i++)
               for (int j=0; j<N; j++)
                    for (int k=0; k<L; k++)
                         dst[ID3(j,k,i,L,M)] =
                              src[ID3(i,j,k,N,L)];
     }
     return b;
}

void ftens_tch(ftens *a, ftens *b)
{
     const int M=a->M, N=a->N, L=a->L, D=a->D;
     for (int w=0; w<D; w++) {
          float *src = a->data + w*a->MNL;
          float *dst = b->data + w*b->MNL;
          for (int i=0; i<M; i++)
               for (int j=0; j<N; j++)
                    for (int k=0; k<L; k++)
                         dst[ID3(j,k,i,L,M)] =
                              src[ID3(i,j,k,N,L)];
     }
}

void ftens_lower(ftens *src, ftens *dst,
                 int W, int H, int Sx, int Sy)
{
     const int D=src->D;
     const int Ms=src->M, Ns=src->N, Ls=src->L;
     const int Md=dst->M, Nd=dst->N, Ld=src->L;
     ASSERT(Ls == Ld && dst->D == D, "err: lowering shape\n");
     float *d = dst->data; int n=0;
     for (int w=0;  w < D; w++) {
          float *s = src->data + w*src->MNL;
          for (int i=0; i<Md; i++)
               for (int j=0; j<Nd; j++)
                    for (int y=0; y<H; y++)
                         for (int x=0; x<W; x++)
                              for (int k=0; k<Ld; k++)
                                   d[n++] =
                                        s[ID3(i*Sy+y,j*Sx+x,k,Ns,Ls)];
     }
}


void ftens_maxpool(ftens *src, ftens *dst, int W, int H,
                   int Sx, int Sy)
{
     const int D =src->D, L =src->L;
     const int Ms=src->M, Ns=src->N;
     const int Md=dst->M, Nd=dst->N;
     ASSERT(D==dst->D && L==dst->L, "err: pool shape\n");
     float *d=dst->data; int n=0;
     for (int w=0; w < D; w++) {
          float *s=src->data + w*src->MNL;
          for (int i=0; i < Ms; i+=Sy)
               for (int j=0; j < Ns; j+=Sx)
                    for (int k=0; k < L; k++) {
                         float v, max=FLT_MIN;
                         for (int y=0; y<H; y++)
                              for (int x=0; x<W; x++) {
                                   v = s[ID3(i+y,j+x,k,Ns,L)];
                                   if (v > max) max = v;
                              }
                         d[n++] = max;
                    }
     }
}


ftens ftens_copy_pad(ftens *t, int p)
{
     const int Ms=t->M, Ns=t->N, L=t->L, D=t->D;
     const int Md=PAD(Ms,p), Nd=PAD(Ns,p);
     ftens out = ftens_zeros(D, Md, Nd, L);
     float *pin  = t->data;
     float *pout = out.data;
     for (int w=0; w < D; w++) {
          for (int i=0; i < Ms; i++)
               for (int j=0; j < Ns; j++)
                    for (int k=0; k < L; k++)
                         pout[ID3(i+p,j+p,k,Nd,L)] =
                              pin[ID3(i,j,k,Ns,L)];
          pin += t->MNL;
          pout += out.MNL;
     }
     return out;
}

void ftens_pad(ftens *src, ftens *dst, int p)
{
     const int D=src->D, L=src->L;
     const int Ms=src->M, Ns=src->N;
     const int Md=dst->M, Nd=dst->N;
     ASSERT(D==dst->D && L==dst->L, "err: pad shape\n");
     float *s = src->data;
     float *d = dst->data;
     memset(d, 0, dst->bytes);
     for (int w=0; w < D; w++) {
          for (int i=0; i < Ms; i++)
               for (int j=0; j < Ns; j++)
                    for (int k=0; k < L; k++)
                         d[ID3(i+p,j+p,k,Nd,L)] =
                              s[ID3(i,j,k,Ns,L)];
          s += src->MNL;
          d += dst->MNL;
     }
}

void ftens_print(ftens *t, const char *fmt)
{
     if (!t->data) {printf("ftens NULL\n"); return;}
     const int M=t->M, N=t->N, L=t->L, D=t->D;
     float *ptr = t->data;
     for (int w=0; w < D; w++) {
          for (int i=0; i < M; i++) {
               for (int k=0; k < L; k++) {
                    for (int j=0; j < N; j++) {
                         float v = ptr[ID3(i,j,k,N,L)];
                         printf(fmt, v);
                    } printf(" | ");
               } NL;
          }
          ptr += t->MNL; NL;
     }
     NL;
}

void ftens_print_ch(ftens *t, int w, int k, int I, int J,
                    const char *fmt)
{
     if (!t->data) {printf("ftens NULL\n"); return;}
     const int D=t->D, M=t->M, N=t->N, L=t->L;
     ASSERT(w < D, "err: print\n");
     float *ptr = t->data + w * t->MNL;
     for (int i=0; i < MIN(M, (unsigned)I); i++) {
          for (int j=0; j < MIN(N,(unsigned)J); j++) {
               printf(fmt, ptr[ID3(i,j,k,N,L)]);
          }
          NL;
     }
}
