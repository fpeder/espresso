static __global__
void ker_bnorm (const float * __restrict__ mean,
                const float * __restrict__ istd,
                const float * __restrict__ beta,
                const float * __restrict__ gamma,
                const int len, const int N,
                float *      __restrict__  dst)
{
     int i=threadIdx.x + blockIdx.x * blockDim.x;

     if (i >= len) return;

     dst[i] = (dst[i] - mean[i%N]) * istd[i%N] * gamma[i%N] + beta[i%N];
}
