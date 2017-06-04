#include "util.cuh"
#include "cumlp.h"



cumlp cumlp_init(int Ndl, int Nbnl)
{
     cumlp out = {Ndl, Nbnl};
     out.dl  = Ndl  ? MALLOC(cudenseLayer, Ndl)  : NULL;
     out.bnl = Nbnl ? MALLOC(cubnormLayer, Nbnl) : NULL;
     for (int i=0; i<Ndl;  i++) DENSEL_INIT(out.dl[i]);
     for (int i=0; i<Nbnl; i++) BNORML_INIT(out.bnl[i]);
     return out;
}


void cumlp_free(cumlp *net)
{
     cuinputLayer_free(&net->il);
     for (int i=0; i<net->Ndl;  i++) cudenseLayer_free(&net->dl[i]);
     for (int i=0; i<net->Nbnl; i++) cubnormLayer_free(&net->bnl[i]);
}


cumlp cumlp_convert(mlp *net)
{
     cumlp out = cumlp_init(net->Ndl, net->Nbnl);
     for (int i=0; i<net->Ndl;  i++)
          cudenseLayer_convert(&net->dl[i], &out.dl[i]);

     for (int i=0; i<net->Nbnl; i++)
          cubnormLayer_convert(&net->bnl[i], &out.bnl[i]);

     return out;
}


void cumlp_print(cumlp *net)
{
     printf("cumlp: Ndl=%d Nbnl=%d\n", net->Ndl, net->Nbnl);
}
