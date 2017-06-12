#include "util.h"
#include "params.h"
#include "nn/mlp.h"


mlp mlp_init(int Ndl, int Nbnl)
{
     mlp out = {Ndl, Nbnl};
     out.dl  = Ndl  ? MALLOC(denseLayer, Ndl)  : NULL;
     out.bnl = Nbnl ? MALLOC(bnormLayer, Nbnl) : NULL;
     for (int i=0; i<Ndl;  i++) DENSEL_INIT(out.dl[i]);
     for (int i=0; i<Nbnl; i++) BNORML_INIT(out.bnl[i]);
     return out;
}

void mlp_free(mlp *net)
{
     inputLayer_free(&net->il);
     for (int i=0; i<net->Ndl;  i++) denseLayer_free(&net->dl[i]);
     for (int i=0; i<net->Nbnl; i++) bnormLayer_free(&net->bnl[i]);
}

mlp mlp_load(const char *esp, int bin)
{
     mlp out;
     int Ndl;  denseLayer *dl;
     int Nbnl; bnormLayer *bnl;

     FILE *pf = fopen(esp, "rb");
     ASSERT(pf, "err: esp fopen\n");

     int val;
     while ((val = fgetc(pf)) != EOF) {
          switch (val) {
          case DENSEL|LNUM: fread(&Ndl,  sizeof(int), 1, pf); break;
          case BNORML|LNUM: fread(&Nbnl, sizeof(int), 1, pf); break;

          case INPUTL|LDAT:
               out = mlp_init(Ndl, Nbnl);
               dl=out.dl; bnl=out.bnl;
               break;

          case DENSEL|LDAT: load_denseLayer(dl, pf, bin); dl++;  break;
          case BNORML|LDAT: load_bnormLayer(bnl, pf);     bnl++; break;
               break;

          default:
               fprintf(stderr, "err: mlp loader\n");
               exit(-3);
          }
     }

     fclose(pf);
     return out;
}

void mlp_print(mlp *net)
{
     printf("mlp: Ndl=%d Nbnl=%d\n", net->Ndl, net->Nbnl);
}
