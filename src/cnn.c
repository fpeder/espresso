#include "util.h"
#include "layers.h"
#include "params.h"
#include "nn/cnn.h"


cnn cnn_init(int Ncl, int Npl, int Ndl, int Nbnl)
{
     cnn out = {Ncl, Npl, Ndl, Nbnl};
     out.cl  = Ncl  ? MALLOC(convLayer,   Ncl) : NULL;
     out.pl  = Npl  ? MALLOC(poolLayer,   Npl) : NULL;
     out.dl  = Ndl  ? MALLOC(denseLayer,  Ndl) : NULL;
     out.bnl = Nbnl ? MALLOC(bnormLayer, Nbnl) : NULL;
     for (int i=0; i<Ncl; i++)  CONVL_INIT(out.cl[i]);
     for (int i=0; i<Npl; i++)  POOLL_INIT(out.pl[i]);
     for (int i=0; i<Ndl; i++)  DENSEL_INIT(out.dl[i]);
     for (int i=0; i<Nbnl; i++) BNORML_INIT(out.bnl[i]);
     return out;
}

cnn cnn_load(const char *esp, int bin, int rev)
{
     cnn out;
     int Ncl;  convLayer  *cl;
     int Npl;  poolLayer  *pl;
     int Ndl;  denseLayer *dl;
     int Nbnl; bnormLayer *bnl;

     FILE *pf = fopen(esp, "rb");
     ASSERT(pf, "err: esp fopen\n");

     int val;
     while ((val = fgetc(pf)) != EOF) {
          switch (val) {
          case CONVL |LNUM: fread(&Ncl,  sizeof(int), 1, pf); break;
          case POOLL |LNUM: fread(&Npl,  sizeof(int), 1, pf); break;
          case DENSEL|LNUM: fread(&Ndl,  sizeof(int), 1, pf); break;
          case BNORML|LNUM: fread(&Nbnl, sizeof(int), 1, pf); break;

          case INPUTL|LDAT:
               out = cnn_init(Ncl, Npl, Ndl, Nbnl);
               cl=out.cl; pl=out.pl; dl=out.dl; bnl=out.bnl;
               break;

          case CONVL |LDAT: load_convLayer(cl, pf, bin, rev); cl++;  break;
          case POOLL |LDAT: load_poolLayer(pl, pf);           pl++;  break;
          case DENSEL|LDAT: load_denseLayer(dl, pf, bin);     dl++;  break;
          case BNORML|LDAT: load_bnormLayer(bnl, pf);         bnl++; break;

          default:
               fprintf(stderr, "err: cnn load\n");
               exit(-3);
          }
     }

     fclose(pf);
     return out;

}

void cnn_print(cnn *net)
{
     printf("CNN: Ncl=%d Npl=%d Ndl=%d Nbnl=%d\n",
            net->Ncl, net->Npl, net->Ndl, net->Nbnl);
}

void cnn_free(cnn *net)
{
     inputLayer_free(&net->il);
     for (int i=0; i<net->Ncl; i++)  convLayer_free(net->cl + i);
     for (int i=0; i<net->Npl; i++)  poolLayer_free(net->pl + i);
     for (int i=0; i<net->Ndl; i++)  denseLayer_free(net->dl + i);
     for (int i=0; i<net->Nbnl; i++) bnormLayer_free(net->bnl + i);
}
