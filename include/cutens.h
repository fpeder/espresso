#ifndef EPS_CUTENS_H
#define EPS_CUTENS_H

#include "util.cuh"
#include "cumem.h"
#include "tens.h"


typedef struct {
     int D, M, N, L, MNL;
     int bytes;
     float *data;
} cuftens;


#define CUFTENS_INIT(t,D,M,N,L) (                     \
          t.D=D, t.M=M, t.N=N, t.L=L, t.MNL=M*N*L,    \
          t.bytes=BYTES(float, D*M*N*L),              \
          t.data=NULL)

#define DIM_CHECK(D,M,N,L)                      \
     cuASSERT(D>0 && M>0 && N>0 && N>0 && L>0,  \
              "err: cuftens invalid size\n")


cuftens cuftens_empty(int D, int M, int N, int L);
cuftens cuftens_init(int D, int M, int N, int L);
cuftens cuftens_from_cufmem(int D, int M, int N, int L);
cuftens cuftens_lower_init(cuftens *t, int W, int H, int Sx, int Sy);
cuftens cuftens_lower_cufmem(cuftens *t, int W, int H, int Sx, int Sy);
cuftens cuftens_zeros(int D, int M, int N, int L);
cuftens cuftens_ones(int D, int M, int N, int L);
cuftens cuftens_rand(int D, int M, int N, int L);
cuftens cuftens_rand(int D, int M, int N, int L,
                     float min, float max);

void    cuftens_round_up(cuftens *src, cuftens *dst);
cuftens cuftens_round_up(ftens *t, int n);
cuftens cuftens_round_up(cuftens *t, int n);
cuftens cuftens_round_up_cufmem(cuftens *t, int n);
cuftens cuftens_copy(cuftens *t);
void    cuftens_copy(cuftens *src, cuftens *dst);
cuftens cuftens_convert(ftens *t);

void    cuftens_pad(cuftens *src, cuftens *dst, int p);
cuftens cuftens_pad(cuftens *t, int p);

ftens cuftens_dump(cuftens *t);

void cuftens_free(cuftens *t);
void cuftens_reshape(cuftens *t, int D, int M, int N, int L);
void cuftens_print_shape(cuftens *t);
void cuftens_print(cuftens *t);
void cuftens_print(cuftens *t, const char *fmt);
void cuftens_print_ch(cuftens *t, int b, int ch, int I, int J,
                      const char *fmt);

static inline
int cuftens_len(cuftens *t) {return t->bytes/sizeof(float);}


#endif /* EPS_CUTENS_H */
