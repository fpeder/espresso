#include "kernels.h"


void cupsignAct_forward(cuftens *src, cuptens *out)
{
     int D=src->D, M=src->M, N=src->N, L=src->L;
     cuASSERT(L % 64 == 0 || N % 64 == 0, "err: psignact % 64\n");

     if (!out->data) *out = cuptens_init(D, M, N, L);
     cupack(src, out);
}
