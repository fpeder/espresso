#include "util.cuh"
#include "pack.cuh"
#include "cuptens.h"

// void cupack(const float *a, uint32_t *b, int M, int N, int L)
// {
//      ker_pack <<<CEIL(M*N*L, 32), 32>>> (a, b);
// }


void cupack(cuftens *src, cuptens *dst)
{
     const int len = cuftens_len(src);
     ker_pack <<<CEIL(len, 32), 32>>>
          (src->data, (uint32_t *)dst->data);
}
