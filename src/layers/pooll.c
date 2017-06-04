#include "util.h"
#include "layers/pool.h"


poolLayer poolLayer_init(int M, int N, int Sm, int Sn)
{
     poolLayer pl = {M, N, Sm, Sn, MAX};
     pl.out. data = NULL;
     pl.mask.data = NULL;
     return pl;
}


void poolLayer_free(poolLayer *pl)
{
     ftens_free(&pl->out);
     ftens_free(&pl->mask);
}


void poolLayer_forward(ftens *t, poolLayer *pl)
{
     const int W=pl->M, H=pl->N, Sy=pl->Sm, Sx=pl->Sn;
     const int D=t->D, L=t->L, Ms=t->M, Ns=t->N;
     const int Md=OUT_LEN(Ms, H, Sy);
     const int Nd=OUT_LEN(Ns, W, Sx);

     if (!pl->out.data) pl->out=ftens_init(D, Md, Nd, L);

     if (pl->op == MAX)
          ftens_maxpool(t, &pl->out, W, H, Sx, Sy);
     else
          exit(-3);
}


void poolLayer_backward(ftens *dout, poolLayer *pl)
{
     exit(-2);
}
