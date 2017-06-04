#include <cblas.h>
#include "util.h"
#include "conv.h"

extern float *scratch;


convLayer convLayer_init(int Sm, int Sn, int p)
{
     convLayer cl; CONVL_INIT(cl);
     cl.Sm=Sm; cl.Sn=Sn; cl.p=p;
     return cl;
}


void convLayer_free(convLayer *cl)
{
     ftens_free(&cl->W);   ftens_free(&cl->b);
     ftens_free(&cl->dW);  ftens_free(&cl->db);
     ftens_free(&cl->out); ftens_free(&cl->in);
}


void convLayer_print_shape(convLayer *cl)
{
     printf("conv: D=%d M=%d N=%d L=%d Sm=%d Sn=%d p=%d\n",
            cl->D, cl->M, cl->N, cl->L, cl->Sm, cl->Sn, cl->p);
}


void convLayer_set(ftens *W, convLayer *cl)
{
     int D=W->D, M=W->M, N=W->N, L=W->L;
     ftens_free(&cl->W);
     cl->D=D; cl->M=M; cl->N=N; cl->L=L;
     cl->W = ftens_copy(W);
}


void convLayer_copy_input(ftens *t, convLayer *cl)
{
     if (!cl->in.data)
          cl->in=ftens_init(t->D, t->M, t->N, t->L);
     memcpy(cl->in.data, t->data, t->bytes);
}


ftens convLayer_pad_input(ftens *t, float *scr,
                          int *M, int *N, int p)
{
     ftens tp; const int D=t->D, L=t->L;
     *M=PAD(*M, p); *N=PAD(*N, p);
     if (!scratch) tp=ftens_copy_pad(t, p);
     else {
          tp = ftens_from_ptr(D, *M, *N, L, scr);
          ftens_pad(t, &tp, p);
          scr += (*M)*(*N)*L*D;
     }

     return tp;
}


void convLayer_forward(ftens *t, convLayer *cl, int save)
{
     float *scr = scratch; ftens tp, tmp;
     int D=t->D,  Ms=t->M, Ns=t->N, Ls=t->L;
     int F=cl->D, W=cl->M, H=cl->N, L=cl->L;
     int p=cl->p, Sy=cl->Sm, Sx=cl->Sn;
     ASSERT(t->L == cl->L, "err: conv shape\n");

     if (save)      convLayer_copy_input(t, cl);
     if (p)    tp = convLayer_pad_input(t, scr, &Ms, &Ns, p);

     // lower
     const int Md = OUT_LEN(Ms, H, Sy);
     const int Nd = OUT_LEN(Ns, W, Sx);
     const int Ld =  W*H*L;
     if (!scratch) tmp=ftens_init(D, Md, Nd, Ld);
     else          tmp=ftens_from_ptr(D, Md, Nd, Ld, scr);

     ftens_lower(p ? &tp : t, &tmp, W, H, Sx, Sy);

     // mat mul
     if (!cl->out.data) cl->out=ftens_init(D, Md, Nd, F);
     int M=Md*Nd, N=F, K=cl->W.MNL;
     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                 M, N, K, 1, tmp.data, K, cl->W.data, K,
                 0, cl->out.data, N);


     if (!scratch)      ftens_free(&tmp);
     if (!scratch && p) ftens_free(&tp);
}


void convLayer_backward(ftens *dout, convLayer *cl)
{
     exit(-2);
}


void convLayer_update(convLayer *cl)
{
     exit(-3);
}
