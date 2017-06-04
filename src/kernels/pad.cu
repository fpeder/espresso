#include "pad.cuh"
#include "cuptens.h"


void cupad(cuftens *src, cuftens *dst, int p)
{
     const int D=src->D, L=src->L;
     const int Ms=src->M, Ns=src->N;
     const int Md=dst->M, Nd=dst->N;

     cuASSERT(D == dst->D &&
              L == dst->L &&
              Md >= Ms    &&
              Nd >= Ns, "err: pad dim\n");

     cudaMemset(dst->data, 0, dst->bytes);

     cupad_template <float>
          (src->data, dst->data, p, D, L, Ms, Ns, Md, Nd);

     // if (L == 1) {
     //      const int BS = 16;
     //      dim3 grid(CEIL(Ns, BS), CEIL(Ms, BS));
     //      dim3 block(BS, BS);
     //      for (int w=0; w < D; w++) {
     //           float *s = src->data + w * src->MNL;
     //           float *d = dst->data + w * dst->MNL;
     //           ker_pad2D <float> <<<grid, block>>>
     //                (s, d, p, Ms, Ns, Md, Nd);
     //      }
     // } else {
     //      const int BS = 8;
     //      dim3 grid(CEIL(L, BS), CEIL(Ns, BS), CEIL(Ms, BS));
     //      dim3 block(BS, BS, BS);
     //      for (int w=0; w < D; w++) {
     //           float *s = src->data + w * src->MNL;
     //           float *d = dst->data + w * dst->MNL;
     //           ker_pad3D <float> <<<grid, block>>>
     //                (s, d, p, Ms, Ns, Md, Nd, L);
     //      }
     // }
}


void cupad(cuptens *src, cuptens *dst, int p)
{
     const int Ms=src->M, Ns=src->N;
     const int Md=dst->M, Nd=dst->N;
     const int L=src->X, D=src->D;

     cuASSERT(L==dst->X && D==dst->D, "err: pad dim\n");

     cudaMemset(dst->data, 0, dst->bytes);

     cupad_template <uint64_t>
          (src->data, dst->data, p, D, L, Ms, Ns, Md, Nd);

     // const int BS = 8;
     // dim3 grid(CEIL(L, BS), CEIL(Ns, BS), CEIL(Ms, BS));
     // dim3 block(BS, BS, BS);

     // for (int w=0; w < D; w++) {
     //      uint64_t *s = src->data + w * src->MNL;
     //      uint64_t *d = dst->data + w * dst->MNL;
     //      ker_pad3D <uint64_t> <<<grid, block>>>
     //           (s, d, p, Ms, Ns, Md, Nd, L);
     // }
}
