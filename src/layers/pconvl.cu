#include "cumem.h"
#include "layers/cuconv.h"
#include "layers/cupconv.h"
#include "kernels.h"


cupconvLayer cupconvLayer_init(int Sm, int Sn, int p)
{
     cupconvLayer out;
     CUPCONVL_INIT(out);
     out.Sm=Sm; out.Sn=Sn; out.p=p;
     return out;
}

void cupconvLayer_free(cupconvLayer *cl)
{
     cuptens_free(&cl->W);  cuptens_free(&cl->pout);
     cuftens_free(&cl->dW); cuftens_free(&cl->out);
     cuftens_free(&cl->in); cuftens_free(&cl->fix);
     cuftens_free(&cl->bfix);
}

void cupconvLayer_set(ftens *W, cupconvLayer *cl, int fix)
{
     int D=W->D, M=W->M, N=W->N, L=W->L;
     cupconvLayer_free(cl);
     cl->D=D; cl->M=M; cl->N=N; cl->L=L;
     cl->W = cuptens_convert(W);
     if (fix) {
          ftens tmp = ftens_init(1, 1, D, 1);
          for (int i = 0; i < D; i++) {
               float *s = W->data + i * W->MNL; ;
               tmp.data[i] = 0.0f;
               for (int j = 0; j < W->MNL; j++)
                    tmp.data[i] += s[j];
          }
          if (!cl->fix.data) cl->fix = cuftens_init(1, 1, D, 1);
          cudaMemcpy(cl->fix.data, tmp.data, tmp.bytes, cuHtoD);
          ftens_free(&tmp);
     }
}

void cupconvLayer_convert(convLayer *src, cupconvLayer *dst, int fix)
{
     dst->Sm=src->Sm; dst->Sn=src->Sn; dst->p=src->p;
     cupconvLayer_set(&src->W, dst, fix);
}

cuptens cupconvLayer_pad_input(cuptens *t, int p)
{
     cuptens tp;
     int M = PAD(t->M, p);
     int N = PAD(t->N, p);
     if (CUPMEM) {
          tp = cuptens_from_cupmem(t->D, M, N, t->L);
          cupad(t, &tp, p);
     } else
          tp = cuptens_pad(t, p);

     return tp;
}

cuptens cupconvLayer_lower_input(cuptens *t, cupconvLayer *cl)
{
     int W=cl->N, H=cl->M, Sx=cl->Sn, Sy=cl->Sm;
     cuptens tl = (CUPMEM  ?
                   cuptens_lower_cupmem(t, W, H, Sx, Sy) :
                   cuptens_lower_init(t, W, H, Sx, Sy));

     cuplower(t, &tl, W, H, Sx, Sy);

     return tl;
}

void cupconvLayer_forward(cuptens *t, cupconvLayer *cl)
{
     cupmem_reset();

     int D=t->D,  M=t->M,  N=t->N,  L=t->X;
     int F=cl->D, p=cl->p, W=cl->N, H=cl->M;
     cuASSERT(L == cl->W.X, "err: cupconv forward\n");

     cuptens tp = !p ? *t : cupconvLayer_pad_input(t, p);
     cuptens tl = cupconvLayer_lower_input(&tp, cl);

     M=tl.M; N=tl.N;

     if (!cl->out.data) cl->out = cuftens_init(D, M, N, F);

     pgemm(D*M*N, F, W*H*L, tl.data, cl->W.data, cl->out.data);

     if(!CUPMEM)      cuptens_free(&tl);
     if(!CUPMEM && p) cuptens_free(&tp);
}

void cupconvLayer_forward_initial(cuftens *t, cupconvLayer *cl,
                                  float norm)
{
     cufmem_reset();
     cupmem_reset();

     cuftens tmp = (!CUFMEM ?
                    cuftens_round_up(t, 64) :
                    cuftens_round_up_cufmem(t, 64));

     int D=t->D,  M=tmp.M, N=tmp.N, L=tmp.L;
     int F=cl->D, p=cl->p, W=cl->N, H=cl->M;

     cuftens tp = !p ? tmp : cuconvLayer_pad_input(&tmp, p);
     cuftens tl = cuconvLayer_lower_input(&tp, (cuconvLayer*)cl);

     M=tl.M; N=tl.N; L=tl.L;

     cuptens qwe = (!CUPMEM ?
                    cuptens_init(D, 8, M*N*L, 1) :
                    cuptens_from_cupmem(D, 8, M*N*L, 1));


     cubp_split_pack(&tl, &qwe);

     cufmem_reset();
     cuftens tmp2 = (!CUFMEM ?
                     cuftens_init(D, M*8, N, F) :
                     cuftens_from_cufmem(D, M*8, N, F));

     pgemm_init_rev(D*M*N*8, F, cl->W.MNL, qwe.data,
                    cl->W.data, tmp2.data);

     if (!cl->out.data) cl->out = cuftens_init(D, M, N, F);
     cubp_merge(&tmp2, &cl->out, &cl->fix, norm);

     if (!CUFMEM)     {cuftens_free(&tmp); cuftens_free(&tl);}
     if (!CUPMEM)      cuptens_free(&qwe);
     if (!CUFMEM && p) cuftens_free(&tp);
}

void cupconvLayer_print(cupconvLayer *cl)
{
     printf("cupconvLayer: \n");
}
