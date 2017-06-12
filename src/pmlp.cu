#include "util.cuh"
#include "nn/cupmlp.h"


cupmlp cupmlp_init(int Ndl, int Nbnl)
{
     cupmlp out = {Ndl, Nbnl};
     out.dl  = Ndl  ? MALLOC(cupdenseLayer, Ndl)  : NULL;
     out.bnl = Nbnl ? MALLOC(cubnormLayer,  Nbnl) : NULL;
     for (int i=0; i < Ndl;  i++) CUPDENSEL_INIT(out.dl[i]);
     for (int i=0; i < Nbnl; i++) BNORML_INIT(out.bnl[i]);
     return out;
}

void cupmlp_free(cupmlp *nn)
{
     cuinputLayer_free(&nn->il);
     for (int i=0; i < nn->Ndl;  i++) cupdenseLayer_free(&nn->dl[i]);
     for (int i=0; i < nn->Nbnl; i++) cubnormLayer_free(&nn->bnl[i]);
}

cupmlp cupmlp_convert(mlp *nn)
{
     cupmlp out = cupmlp_init(nn->Ndl, nn->Nbnl);
     for (int i=0; i < nn->Ndl; i++)
          cupdenseLayer_convert(&nn->dl[i], &out.dl[i], i==0);

     for (int i=0; i < nn->Nbnl; i++)
          cubnormLayer_convert(&nn->bnl[i], &out.bnl[i]);

     return out;
}

void cupmlp_print(cupmlp *nn)
{
     printf("cupmlp: Npdl=%d Nbnl=%d\n", nn->Ndl, nn->Nbnl);
}
