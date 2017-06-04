#include "util.cuh"
#include "kernels.h"
#include "layers/cupool.h"


cupoolLayer cupoolLayer_init(int M, int N, int Sm, int Sn)
{
     cupoolLayer pl = {M, N, Sm, Sn, MAX};
     pl.out. data = NULL;
     pl.mask.data = NULL;
     return pl;
}

void cupoolLayer_free(cupoolLayer *pl)
{
     cuftens_free(&pl->out);
     cuftens_free(&pl->mask);
}

void cupoolLayer_convert(poolLayer *src, cupoolLayer *dst)
{
     dst->M  = src->N;
     dst->N  = src->N;
     dst->Sm = src->Sm;
     dst->Sn = src->Sn;
     dst->op = src->op;
}

void cupoolLayer_forward(cuftens *t, cupoolLayer *pl)
{
     int W=pl->M, H=pl->N, Sy=pl->Sm, Sx=pl->Sn;
     int M = OUT_LEN(t->M, H, Sy);
     int N = OUT_LEN(t->N, W, Sx);

     if (!pl->out.data)
          pl->out = cuftens_init(t->D, M, N, t->L);

     cuASSERT(pl->op == MAX, "err: pool type not impl\n");
     cumaxpool(t, &pl->out, W, H, Sx, Sy);
}


void cupoolLayer_backward(cuftens *dt, cupoolLayer *pl)
{
     exit(-2);
}


void cupoolLayer_print(cupoolLayer *pl)
{
     printf("cupool: %d %d %d %d\n", pl->M, pl->N, pl->Sm, pl->Sn);
}
