#include <string.h>
#include <cblas.h>

#include "util.h"
#include "dense.h"


denseLayer denseLayer_init(int M, int N)
{
     denseLayer dl; DENSEL_INIT(dl); dl.M=M; dl.N=N;
     return dl;
}


void denseLayer_free(denseLayer *dl)
{
     ftens_free(&dl->W);   ftens_free(&dl->b);
     ftens_free(&dl->dW);  ftens_free(&dl->db);
     ftens_free(&dl->out); ftens_free(&dl->in);
}


void denseLayer_print_shape(denseLayer *dl)
{
     printf("dense: %d %d\n", dl->M, dl->N);
}


void denseLayer_set(ftens *W, denseLayer *dl)
{
     const int M=W->M, N=W->N;
     ASSERT(W->D==1 && W->L==1, "err: dense shape\n");
     ftens_free(&dl->W);
     dl->M = M; dl->N = N;
     dl->W = ftens_copy(W);
}


void denseLayer_forward(ftens *t, denseLayer *dl, int save)
{
     const int D=t->D, M=dl->M, N=dl->N;
     ASSERT(t->MNL == dl->N,  "err: dense shape\n");

     if (save) {
          int M=t->M, N=t->N, L=t->L;
          if (!dl->in.data) dl->in = ftens_init(D,M,N,L);
          memcpy(dl->in.data, t->data, t->bytes);
     }

     if (!dl->out.data) dl->out = ftens_init(D, 1, M, 1);
     const float *a=dl->W.data;
     const float *b=t->data;
     float       *c=dl->out.data;

     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                 D, M, N, 1, b, N, a, N, 0, c, M);

}


void denseLayer_backward(ftens *dout, denseLayer *dl)
{
     fprintf(stderr, "not implemented yet\n");
     exit(-2);
}
