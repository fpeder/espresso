#include "cutens.h"
#include "kernels.h"


cuftens cuftens_empty(int D, int M, int N, int L)
{
     cuftens t;
     DIM_CHECK(D, M, N, L);
     CUFTENS_INIT(t, D, M, N, L);
     return t;
}

cuftens cuftens_init(int D, int M, int N, int L)
{
     cuftens t = cuftens_empty(D, M, N, L);
     cudaMalloc(&t.data, t.bytes);
     cuASSERT(t.data, "err: cuftens cuMalloc\n");
     return t;
}

cuftens cuftens_lower_init(cuftens *t, int W, int H, int Sx, int Sy)
{
     int M = OUT_LEN(t->M, H, Sy);
     int N = OUT_LEN(t->N, W, Sx);
     return cuftens_init(t->D, M, N, W*H*t->L);
}

cuftens cuftens_lower_cufmem(cuftens *t, int W, int H, int Sx, int Sy)
{
     int M = OUT_LEN(t->M, H, Sy);
     int N = OUT_LEN(t->N, W, Sx);
     cuftens out = cuftens_empty(t->D, M, N, W*H*t->L);
     out.data    = cufmem_reserve(out.bytes);
     return out;
}

void cuftens_free(cuftens *t)
{
     if (t->data) {cudaFree(t->data); t->data=NULL;}
}

cuftens cuftens_zeros(int D, int M, int N, int L)
{
     cuftens t = cuftens_init(D, M, N, L);
     cudaMemset(t.data, 0, t.bytes);
     return t;
}

cuftens cuftens_ones(int D, int M, int N, int L)
{
     cuftens t = cuftens_init(D, M, N, L);
     cuset(&t, 1.0);
     return t;
}

cuftens cuftens_rand(int D, int M, int N, int L)
{
     cuftens t = cuftens_init(D, M, N, L);
     ftens tmp = ftens_rand(D, M, N, L);
     cudaMemcpy(t.data, tmp.data, t.bytes, cuHtoD);
     ftens_free(&tmp);
     return t;
}

cuftens cuftens_rand(int D, int M, int N, int L, float min, float max)
{
     cuftens t = cuftens_init(D, M, N, L);
     ftens tmp = ftens_rand_range(D, M, N, L, min, max);
     cudaMemcpy(t.data, tmp.data, t.bytes, cuHtoD);
     ftens_free(&tmp);
     return t;
}

cuftens cuftens_from_cufmem(int D, int M, int N, int L)
{
     cuftens t = cuftens_empty(D, M, N, L);
     t.data    = cufmem_reserve(t.bytes);
     return t;
}

cuftens cuftens_convert(ftens *t)
{
     cuftens out = cuftens_init(t->D, t->M, t->N, t->L);
     cudaMemcpy(out.data, t->data, t->bytes, cuHtoD);
     return out;
}

void cuftens_reshape(cuftens *t, int D, int M, int N, int L)
{
     const int len = cuftens_len(t);
     cuASSERT(len == D*M*N*L, "err: cuftens reshape\n");
     t->D=D; t->M=M; t->N=N; t->L=L;
}

cuftens cuftens_copy(cuftens *t)
{
     cuASSERT(t->data, "err\n");
     cuftens out = cuftens_init(t->D, t->M, t->N, t->L);
     cudaMemcpy(out.data, t->data, t->bytes, cuDtoD);
     return out;
}

void cuftens_copy(cuftens *src, cuftens *dst)
{
     cuASSERT(src->data && dst->data, "err\n");
     cucopy(src, dst);
}

void cuftens_pad(cuftens *src, cuftens *dst, int p)
{
     cuASSERT(dst->data && dst->data, "err\n");
     cupad(src, dst, p);
}

cuftens cuftens_pad(cuftens *t, const int p)
{
     const int D=t->D, M=t->M, N=t->N, L=t->L;
     cuftens out = cuftens_zeros(D, PAD(M,p), PAD(N,p), L);
     cupad(t, &out, p);
     return out;
}

cuftens cuftens_round_up(ftens *t, int n)
{
     cuftens tmp = cuftens_convert(t);
     cuftens out = cuftens_round_up(&tmp, n);
     cuftens_free(&tmp);
     return out;
}

cuftens cuftens_round_up(cuftens *t, int n)
{
     int D=t->D, M=t->M, N=t->N, L=t->L;
     if (L > 1) L = ROUND_UP(L, n);
     else       N = ROUND_UP(N, n);
     cuftens out = cuftens_zeros(D, M, N, L);
     cucopy(t, &out);
     return out;
}

cuftens cuftens_round_up_cufmem(cuftens *t, int n)
{
     int D=t->D, M=t->M, N=t->N, L=t->L;
     if (L > 1) L = ROUND_UP(L, n);
     else       N = ROUND_UP(N, n);
     cuftens out = cuftens_from_cufmem(D, M, N, L);
     cudaMemset(out.data, 0, out.bytes);
     cucopy(t, &out);
     return out;
}

ftens cuftens_dump(cuftens *t)
{
     int D=t->D, M=t->M, N=t->N, L=t->L;
     cuASSERT(t->data, "err\n");
     ftens out = ftens_init(D, M, N, L);
     cudaMemcpy(out.data, t->data, t->bytes, cuDtoH);
     return out;
}

void cuftens_print(cuftens *t)
{
     cuftens_print(t, "%.2f ");
}

void cuftens_print(cuftens *t, const char *fmt)
{
     if (!t->data) {printf("tens null\n"); return;}
     ftens tmp = cuftens_dump(t);
     ftens_print(&tmp, fmt);
     ftens_free(&tmp);
}

void cuftens_print_ch(cuftens *t, int b, int ch, int I, int J,
                      const char *fmt)
{
     ftens tmp = cuftens_dump(t);
     ftens_print_ch(&tmp, b, ch, I, J, fmt);
     ftens_free(&tmp);
}

void cuftens_print_shape(cuftens *t)
{
     printf("cuftens: %d %d %d %d\n", t->D, t->M, t->N, t->L);
}
