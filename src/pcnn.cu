#include "util.cuh"
#include "nn/cupcnn.h"


cupcnn cupcnn_init(int Ncl, int Npl, int Ndl, int Nbnl)
{
     cupcnn out = {Ncl, Npl, Ndl, Nbnl};
     out.cl  = Ncl  ? MALLOC(cupconvLayer,  Ncl)  : NULL;
     out.pl  = Npl  ? MALLOC(cupoolLayer,   Npl)  : NULL;
     out.dl  = Ndl  ? MALLOC(cupdenseLayer, Ndl)  : NULL;
     out.bnl = Nbnl ? MALLOC(cubnormLayer,  Nbnl) : NULL;
     for (int i=0; i<Ncl;  i++) CUPCONVL_INIT (out.cl[i]);
     for (int i=0; i<Npl;  i++) POOLL_INIT    (out.pl[i]);
     for (int i=0; i<Ndl;  i++) CUPDENSEL_INIT(out.dl[i]);
     for (int i=0; i<Nbnl; i++) BNORML_INIT   (out.bnl[i]);
     return out;
}

void cupcnn_free(cupcnn *net)
{
     cuinputLayer_free(&net->il);
     for (int i=0; i<net->Ncl;  i++) cupconvLayer_free(net->cl + i);
     for (int i=0; i<net->Npl;  i++) cupoolLayer_free(net->pl + i);
     for (int i=0; i<net->Ndl;  i++) cupdenseLayer_free(net->dl + i);
     for (int i=0; i<net->Nbnl; i++) cubnormLayer_free(net->bnl + i);
}

cupcnn cupcnn_convert(cnn *net)
{
     int Ncl=net->Ncl, Npl=net->Npl;
     int Ndl=net->Ndl, Nbnl=net->Nbnl;

     cupcnn out = cupcnn_init(Ncl, Npl, Ndl, Nbnl);

     for (int i=0; i<Ncl; i++)
          cupconvLayer_convert(&net->cl[i], &out.cl[i], i==0);

     for (int i=0; i<Npl; i++)
          cupoolLayer_convert(&net->pl[i], &out.pl[i]);

     for (int i=0; i<Ndl; i++)
          cupdenseLayer_convert(&net->dl[i], &out.dl[i], i==0);

     for (int i=0; i<Nbnl; i++)
          cubnormLayer_convert(&net->bnl[i], &out.bnl[i]);

     return out;
}

void cupcnn_print(cupcnn *net)
{
     printf("CUPCNN: Ncl=%d Npl=%d Ndl=%d Nbnl=%d\n",
            net->Ncl, net->Npl, net->Ndl, net->Nbnl);
}
