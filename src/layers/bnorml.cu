#include "util.cuh"
#include "layers/cubnorm.h"
#include "kernels.h"


cubnormLayer cubnormLayer_init(int use_global)
{
     cubnormLayer bnl; BNORML_INIT(bnl);
     bnl.ug = use_global;
     return bnl;
}


void cubnormLayer_free(cubnormLayer *bnl)
{
     cuftens_free(&bnl->mean);  cuftens_free(&bnl->istd);
     cuftens_free(&bnl->gmean); cuftens_free(&bnl->gistd);
     cuftens_free(&bnl->beta);  cuftens_free(&bnl->gamma);
     cuftens_free(&bnl->dbeta); cuftens_free(&bnl->dgamma);
     cuftens_free(&bnl->tmp);   cuftens_free(&bnl->in);
}


void cubnormLayer_print_shape(cubnormLayer *bnl)
{
     printf("cubnorm: N=%d ug=%d\n", bnl->N, bnl->ug);
}


void cubnormLayer_convert(bnormLayer *src, cubnormLayer *dst)
{
     cubnormLayer_set(&src->mean, &src->istd, &src->gamma,
                      &src->beta, dst);
}


void cubnormLayer_set(ftens *mean,  ftens *istd,
                      ftens *gamma, ftens *beta,
                      cubnormLayer *bnl)
{
     const int N=ftens_len(mean);
     cuASSERT(N == ftens_len(istd) &&
              N == ftens_len(beta) &&
              N == ftens_len(gamma), "err: cubnorm shape\n");

     cubnormLayer_free(bnl);
     bnl->N     = N;
     bnl->mean  = cuftens_convert(mean);
     bnl->istd  = cuftens_convert(istd);
     bnl->beta  = cuftens_convert(beta);
     bnl->gamma = cuftens_convert(gamma);
}


void cubnormLayer_forward(cuftens *t, cubnormLayer *bnl, int save)
{
     const int D=t->D, M=t->M, N=t->N, L=t->L;
     if (save) {
          if (!bnl->in.data) bnl->in=cuftens_init(D, M, N, L);
          cudaMemcpy(bnl->in.data, t->data, t->bytes, cuDtoD);
     }

     if (bnl->ug) {
          // compute bath mean, istd
          // moving avg -> update globals
          fprintf(stderr, "not implemented yet\n");
          exit(-3);
     }

     cubnorm(&bnl->mean, &bnl->istd, &bnl->beta, &bnl->gamma, t);
}


void cubnormLayer_backward(cuftens *dout, cubnormLayer *bnl)
{
     fprintf(stderr, "not implemented yet\n");
     exit(-2);
}



void cubnormLayer_update(cubnormLayer *bnl)
{
     fprintf(stderr, "not implemented yet\n");
     exit(-2);
}
