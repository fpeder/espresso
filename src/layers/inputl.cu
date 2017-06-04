#include "util.cuh"
#include "cuinput.h"
#include "kernels.h"


void cuinputLayer_forward(ftens *t, cuinputLayer *il, int norm)
{
     int D=t->D, M=t->M, N=t->N, L=t->L;

     if (!il->out.data) il->out=cuftens_init(D, M, N, L);
     cudaMemcpy(il->out.data, t->data, t->bytes, cuHtoD);

     if (norm) cunorm(&il->out);
}

void cuinputLayer_free(cuinputLayer *il)
{
     cuftens_free(&il->out);
}
