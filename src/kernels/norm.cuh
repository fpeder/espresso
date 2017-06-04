#include "util.h"

static __global__
void ker_norm(float * __restrict__ src, const int len)
{
     int i=threadIdx.x + blockIdx.x * blockDim.x;
     if (i >= len) return;
     src[i] = 2.0f * src[i] / 255.0f - 1.0f;
}
