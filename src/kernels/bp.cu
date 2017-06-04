#include "util.cuh"
#include "bp.cuh"
#include "cuptens.h"


void cubp_split_pack(cuftens *src, cuptens *dst)
{
     int D=src->D, Ns=src->MNL, Nd=dst->X*2;

     dim3 grid(D, CEIL(Ns, 32));
     dim3 block(1, 32);

     ker_bpsplit <8> <<<grid, block>>>
          (src->data, (uint32_t *)dst->data, Ns, Nd);

}

void cubp_merge(cuftens *src, cuftens *dst, cuftens *fix, float norm)
{
     const int D=src->D, N=src->MNL/8;

     dim3 grid(D, CEIL(N, 32));
     dim3 block(8, 32);

     ker_bpmerge <8> <<<grid, block>>>
          (src->data, dst->data, fix->data, norm, N, fix->N);
}
