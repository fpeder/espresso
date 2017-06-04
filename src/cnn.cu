#include "util.cuh"
#include "culayers.h"
#include "cucnn.h"


cucnn cucnn_init(int Ncl, int Npl, int Ndl, int Nbnl)
{
     cucnn out = {Ncl, Npl, Ndl, Nbnl};
     out.cl  = Ncl  ? MALLOC(cuconvLayer,  Ncl)  : NULL;
     out.pl  = Npl  ? MALLOC(cupoolLayer,  Npl)  : NULL;
     out.dl  = Ndl  ? MALLOC(cudenseLayer, Ndl)  : NULL;
     out.bnl = Nbnl ? MALLOC(cubnormLayer, Nbnl) : NULL;
     for (int i=0; i<Ncl;  i++) CONVL_INIT(out.cl[i]);
     for (int i=0; i<Npl;  i++) POOLL_INIT(out.pl[i]);
     for (int i=0; i<Ndl;  i++) DENSEL_INIT(out.dl[i]);
     for (int i=0; i<Nbnl; i++) BNORML_INIT(out.bnl[i]);
     return out;
}

void cucnn_free(cucnn *net)
{
     cuinputLayer_free(&net->il);
     for (int i=0; i<net->Ncl;  i++) cuconvLayer_free(net->cl + i);
     for (int i=0; i<net->Npl;  i++) cupoolLayer_free(net->pl + i);
     for (int i=0; i<net->Ndl;  i++) cudenseLayer_free(net->dl + i);
     for (int i=0; i<net->Nbnl; i++) cubnormLayer_free(net->bnl + i);
}


cucnn cucnn_convert(cnn *net)
{
     const int Ncl=net->Ndl, Npl=net->Npl;
     const int Ndl=net->Ndl, Nbnl=net->Nbnl;
     cucnn out = cucnn_init(Ncl, Npl, Ndl, Nbnl);
     for (int i=0; i<Ncl; i++)
          cuconvLayer_convert(&net->cl[i], &out.cl[i]);

     for (int i=0; i<Npl; i++)
          cupoolLayer_convert(&net->pl[i], &out.pl[i]);

     for (int i=0; i<Ndl; i++)
          cudenseLayer_convert(&net->dl[i], &out.dl[i]);

     for (int i=0; i<Nbnl; i++)
          cubnormLayer_convert(&net->bnl[i], &out.bnl[i]);

     return out;
}


void cucnn_print(cucnn *net)
{
     printf("CUCNN: Ncl=%d Npl=%d Ndl=%d Nbnl=%d\n",
            net->Ncl, net->Npl, net->Ndl, net->Nbnl);
}
