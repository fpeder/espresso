#include "lower.cuh"
#include "cuptens.h"

#define KER_SETUP(type, TM, TN, TL)                                \
     int bytes = (TM+H-1) * (TN+W-1)* TL * sizeof(type);           \
     dim3 grid(CEIL(L, TL), CEIL(Nd, TN), CEIL(Md, TM));           \
     dim3 block(TL, TN, TM)



void culower(cuftens *src, cuftens *dst, int W, int H, int Sx, int Sy)
{
     const int D=src->D,  L=src->L;
     const int Ms=src->M, Ns=src->N;
     const int Md=dst->M, Nd=dst->N;

     cuASSERT(D==dst->D, "err: lower shape\n");

     const int TM=8, TN=8, TL=8;
     KER_SETUP(float, TM, TN, TL);
     for (int w=0; w < D; w++) {
           float *s = src->data + w * src->MNL;
           float *d = dst->data + w * dst->MNL;
           ker_lower <TM, TN, TL> <<<grid, block, bytes>>>
                (s, d, Ms, Ns, Md, Nd, L, W, H, Sx, Sy);
      }
}


void cuplower(cuptens *src, cuptens *dst, int W, int H, int Sx, int Sy)
{
     const int D=src->D,  L=src->X;
     const int Ms=src->M, Ns=src->N;
     const int Md=dst->M, Nd=dst->N;

     cuASSERT(D==dst->D, "err: lower shape\n");

     const int TM=16, TN=16, TL=4;
     KER_SETUP(uint64_t, TM, TN, TL);

     for (int w=0; w < D; w++) {
          uint64_t *s = src->data + w * src->MNL;
          uint64_t *d = dst->data + w * dst->MNL;
          ker_plower <TM, TN, TL> <<<grid, block, bytes>>>
               (s, d, Ms, Ns, Md, Nd, L, W, H, Sx, Sy);
     }
}


#undef KER_SETUP
