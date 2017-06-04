#include "util.cuh"
#include "cudense.h"
#include "kernels.h"


cudenseLayer cudenseLayer_init(int M, int N)
{
     cudenseLayer dl; DENSEL_INIT(dl); dl.M=M; dl.N=N;
     return dl;
}

void cudenseLayer_free(cudenseLayer *dl)
{
     cuftens_free(&dl->W);   cuftens_free(&dl->b);
     cuftens_free(&dl->dW);  cuftens_free(&dl->db);
     cuftens_free(&dl->out); cuftens_free(&dl->in);
}

void cudenseLayer_convert(denseLayer *src, cudenseLayer *dst)
{
     cudenseLayer_set(&src->W, dst);
}

void cudenseLayer_set(ftens *W, cudenseLayer *dl)
{
     int M=W->M, N=W->N;
     cudenseLayer_free(dl);
     dl->M=M; dl->N=N; dl->W=cuftens_init(1, M, N, 1);
     cudaMemcpy(dl->W.data, W->data, W->bytes, cuHtoD);
}

void cudenseLayer_copy_input(cuftens *t, cudenseLayer *dl)
{
     if (!dl->in.data)
          dl->in = cuftens_init(t->D, t->M, t->N, t->L);
     cudaMemcpy(dl->in.data, t->data, t->bytes, cuHtoD);
}

void cudenseLayer_forward(cuftens *t, cudenseLayer *dl, int save)
{
     int D=t->D, M=t->M, N=t->N;
     cuftens_reshape(t, D, 1, t->MNL, 1);
     cuASSERT(t->MNL == dl->N, "err: cudense shape\n");

     if (save)          cudenseLayer_copy_input(t, dl);
     if (!dl->out.data) dl->out=cuftens_init(D, 1, dl->M, 1);

     if (D == 1) sgemv(&dl->W, t, &dl->out);
     else        sgemm(M, 1, N, t->data, N, dl->W.data, N,
                       dl->out.data, 1);
}


void cudenseLayer_backward(cuftens *dt, cudenseLayer *dl)
{
     fprintf(stderr, "err: dense bprop not implemented yet\n");
     exit(-2);
}


void cudenseLayer_print_size(cudenseLayer *dl)
{
     printf("cudense: %d %d\n", dl->M, dl->N);
}
