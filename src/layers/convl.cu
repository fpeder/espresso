#include "util.cuh"
#include "cuconv.h"
#include "kernels.h"


extern cufmem fptr;


cuconvLayer cuconvLayer_init(int Sm, int Sn, int p)
{
     cuconvLayer cl; CONVL_INIT(cl);
     cl.Sm=Sm; cl.Sn=Sn; cl.p=p;
     return cl;
}

void cuconvLayer_free(cuconvLayer *cl)
{
     cuftens_free(&cl->W);   cuftens_free(&cl->b);
     cuftens_free(&cl->dW);  cuftens_free(&cl->db);
     cuftens_free(&cl->out); cuftens_free(&cl->in);
}

void cuconvLayer_set(ftens *W, cuconvLayer *cl)
{
     int D=W->D, M=W->M, N=W->N, L=W->L;
     cuftens_free(&cl->W);
     cl->D=D; cl->M=M; cl->N=N; cl->L=L;
     cl->W = cuftens_init(D, M, N, L);
     cudaMemcpy(cl->W.data, W->data, W->bytes, cuHtoD);
}

void cuconvLayer_convert(convLayer *src, cuconvLayer *dst)
{
     cuconvLayer_set(&src->W, dst);
}

void cuconvLayer_copy_input(cuftens *t, cuconvLayer *cl)
{
     int D=t->D, M=t->M, N=t->N, L=t->L;
     if (!cl->in.data) cl->in = cuftens_init(D, M, N, L);
     cudaMemcpy(cl->in.data, t->data, t->bytes, cuDtoD);
}

cuftens cuconvLayer_pad_input(cuftens *t, int p)
{
     cuftens tp;
     int M = PAD(t->M, p);
     int N = PAD(t->N, p);
     if (!CUFMEM) tp = cuftens_pad(t, p);
     else {tp = cuftens_from_cufmem(t->D, M, N, t->L);
                cuftens_pad(t, &tp, p);}
     return tp;
}

cuftens cuconvLayer_lower_input(cuftens *t, cuconvLayer *cl)
{
     int W=cl->N, H=cl->M, Sx=cl->Sn, Sy=cl->Sm;
     cuftens tl = (CUFMEM ?
                   cuftens_lower_cufmem(t, W, H, Sx, Sy) :
                   cuftens_lower_init(t, W, H, Sx, Sy));

     culower(t, &tl, W, H, Sx, Sy);

     return tl;
}

void cuconvLayer_forward(cuftens *t, cuconvLayer *cl, int save)
{
     cufmem_reset();

     int D=t-> D, M=t->M, N=t->N, L=t->L;
     int F=cl->D, W=cl->N, H=cl->M;
     int p=cl->p, Sx=cl->Sn, Sy=cl->Sm;

     cuASSERT(t->L == cl->L, "err: cuconv shape\n");

     cuftens tp = !p ? *t : cuconvLayer_pad_input(t, p);
     cuftens tl = cuconvLayer_lower_input(&tp, cl);

     M = tl.M; N = tl.N;
     if (!cl->out.data) cl->out=cuftens_init(D, M, N, F);

     M=D*M*N; N=F; int K=W*H*L;
     sgemm(M, N, K, tl.data, K, cl->W.data, K, cl->out.data, N);

     if (!CUFMEM)      cuftens_free(&tl);
     if (!CUFMEM && p) cuftens_free(&tp);
}


void cuconvLayer_backward(cuftens *dout, cuconvLayer *cl)
{
     exit(-2);
}


void cuconvLayer_print_shape(cuconvLayer *cl)
{
     printf("cuconv: D=%d M=%d N=%d L=%d Sm=%d Sn=%d p=%d\n",
            cl->D, cl->M, cl->N, cl->L, cl->Sm, cl->Sn, cl->p);
}
