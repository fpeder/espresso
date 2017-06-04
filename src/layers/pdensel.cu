#include "util.cuh"
#include "kernels.h"
#include "layers/cupdense.h"


cupdenseLayer cupdenseLayer_init()
{
     cupdenseLayer dl;
     CUPDENSEL_INIT(dl);
     return dl;
}

void cupdenseLayer_free(cupdenseLayer *dl)
{
     cuptens_free(&dl->W);  cuptens_free(&dl->pout);
     cuptens_free(&dl->in); cuftens_free(&dl->out);
     cuftens_free(&dl->dW); cuftens_free(&dl->fix);
}

void cupdenseLayer_set(ftens *W, cupdenseLayer *dl, int fix)
{
     int M = W->M, N = W->N;
     cupdenseLayer_free(dl);
     dl->M=M; dl->N=N;  dl->W=cuptens_convert(W);

     if (fix) {
          ftens tmp = ftens_init(1, 1, M, 1);
          for (int i=0; i<M; i++) {
               tmp.data[i] = 0.0f;
               for (int j=0; j < N; j++)
                    tmp.data[i] += W->data[ID2(i,j,N)];
          }
          dl->fix = cuftens_convert(&tmp);
          ftens_free(&tmp);
     }
}

void cupdenseLayer_convert(denseLayer *src, cupdenseLayer *dst, int fix)
{
     cupdenseLayer_set(&src->W, dst, fix);
}

static
void cupdenseLayer_copy_input(cuptens *t, cupdenseLayer *dl)
{
     int D=t->D, M=t->M, N=t->N, L=t->L;
     if (!dl->in.data) dl->in = cuptens_init(D, M, N, L);
     cudaMemcpy(dl->in.data, t->data, t->bytes, cuDtoD);
}

void cupdenseLayer_forward(cuptens *t, cupdenseLayer *dl, int save)
{
     int D=t->D, M=dl->M, N=dl->W.X;
     cuASSERT(t->MNL == dl->N/64, "err: cupdense shape\n");

     if (save) cupdenseLayer_copy_input(t, dl);

     if (!dl->out.data) dl->out = cuftens_init(D, 1, M, 1);
     if (D == 1) pgemv(M, N, dl->W.data, t->data, dl->out.data);
     else        pgemm(D, M, N, t->data, dl->W.data, dl->out.data);
}

static
cuptens cupdenseLayer_bpsplit_input(cuftens *t, cupdenseLayer *dl)
{
     int D=t->D, N=t->MNL, ru=N & 63;
     cuftens tmp; tmp.data=NULL;
     if (ru) {
          if (CUFMEM) {
               int asd = ROUND_UP(N, 64);
               tmp = cuftens_from_cufmem(D, 1, asd, 1);
               cudaMemset(tmp.data, 0, tmp.bytes);
               cucopy(t, &tmp);
          }
          else {
               tmp = cuftens_round_up(t, 64);
          }
     }

     cuptens out = (CUPMEM ?
                    cuptens_from_cupmem(D, 8, N, 1) :
                    cuptens_init(D, 8, N, 1));

     cubp_split_pack(ru ? &tmp : t, &out);

     if (ru && !CUFMEM) cuftens_free(&tmp);
     return out;
}


void cupdenseLayer_forward_initial(cuftens *t, cupdenseLayer *dl,
                                   float norm)
{
     int D=t->D, M=dl->M, N=8, K=dl->W.X;

     cufmem_reset();
     cupmem_reset();

     cuftens_reshape(t, t->D, 1, t->MNL, 1);
     cuptens asd = cupdenseLayer_bpsplit_input(t, dl);
     cuftens tmp = (CUFMEM ?
                    cuftens_from_cufmem(D, N, M, 1) :
                    cuftens_init(D, N, M, 1));

     pgemm_init_rev(D*N, M, K, asd.data, dl->W.data, tmp.data);

     if (!dl->out.data) dl->out = cuftens_init(D, 1, M, 1);

     cubp_merge(&tmp, &dl->out, &dl->fix, norm);

     if (!CUPMEM) cuptens_free(&asd);
     if (!CUFMEM) cuftens_free(&tmp);
}


void cupdenseLayer_backward(cuptens *dout, cupdenseLayer *dl)
{
     fprintf(stderr, "err: cupdensel bprop not yet implemented\n");
     exit(-2);
}

void cupdenseLayer_update(cupdenseLayer *dl)
{
     fprintf(stderr, "err: cupdensel bprop not yet implemented\n");
     exit(-2);
}
