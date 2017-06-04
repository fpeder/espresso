#include "util.cuh"
#include "bnorm.cuh"
#include "cutens.h"

void cubnorm (cuftens *mean, cuftens *istd,
              cuftens *beta, cuftens *gamma,
              cuftens  *in)
{
     const int TS=32;
     const int M=in->N, N=in->N, L=in->L;
     const int len = cuftens_len(in);

     ker_bnorm <<<CEIL(len, TS), TS>>>
          (mean->data, istd->data,
           beta->data, gamma->data, len, L > 1 ? L : M*N,
           in->data);
}
