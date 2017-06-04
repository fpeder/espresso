#include <stdint.h>

static __global__
void ker_pack(const float * __restrict__ a,
              uint32_t *    __restrict__ b)
{
     int i=threadIdx.x + blockIdx.x*blockDim.x;
     b[i>>5] = __ballot((uint32_t)(a[i] > 0.0f));
}
