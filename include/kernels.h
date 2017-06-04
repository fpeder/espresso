#ifndef ESP_KERNELS_H
#define ESP_KERNELS_H

#include <stdint.h>
#include "cuptens.h"


void cuset(cuftens *t, float v);
void cucopy(cuftens *src, cuftens *dst);
void cupad(cuftens *src, cuftens *dst, int p);
void cupad(cuptens *src, cuptens *dst, int p);
void cupack(cuftens *src, cuptens *dst);
void cutch(cuftens *src, cuftens *dst);
void cusign(cuftens *a);
void cunorm(cuftens *t);

void cubp_split_pack(cuftens *src, cuptens *dst);
void cubp_merge(cuftens *src, cuftens *dst, cuftens *fix, float norm);

void culower  (cuftens *src, cuftens *dst, int W, int H, int Sx, int Sy);
void cuplower (cuptens *src, cuptens *dst, int W, int H, int Sx, int Sy);
void cumaxpool(cuftens *src, cuftens *dst, int W, int H, int Sx, int Sy);

void cubnorm(cuftens *mean, cuftens *istd,
             cuftens *beta, cuftens *gamma,
             cuftens  *in);

void sgemv(cuftens *a, cuftens *b, cuftens *c);
void sgemm(cuftens *a, cuftens *b, cuftens *c);

void sgemm(int M, int N, int K,
           const float * __restrict__ A, int lda,
           const float * __restrict__ B, int ldb,
           float *       __restrict__ C, int ldc);

void pgemv(int m, int n,
           const uint64_t * __restrict__ A,
           const uint64_t * __restrict__ x,
           float *          __restrict__ y);

void pgemm(int M, int N, int K,
           const uint64_t * __restrict__ A,
           const uint64_t * __restrict__ B,
           float *          __restrict__ C);

void pgemm_init(int M, int N, int K,
                const uint64_t * __restrict__ A,
                const uint64_t * __restrict__ B,
                float *          __restrict__ C);

void pgemm_init(int M, int N, int K,
                const uint64_t * __restrict__ a, int lda,
                const uint64_t * __restrict__ b, int ldb,
                float *          __restrict__ c, int ldc);

void pgemm_init_rev(int M, int N, int K,
                    const uint64_t * __restrict__ A,
                    const uint64_t * __restrict__ B,
                    float *          __restrict__ C);

void pgemm32(int M, int N, int K,
             const uint32_t * __restrict__ A,
             const uint32_t * __restrict__ B,
             float *          __restrict__ C);



#endif /* ESP_KERNELS_H */
