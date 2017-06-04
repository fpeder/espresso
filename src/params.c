#include "util.h"
#include "layers.h"
#include "mlp.h"
#include "cnn.h"


static inline
void reverse(float *v, const int len)
{
     float t; int j=len-1;
     for (int i = 0; i < len/2; i++) {
          t = v[i]; v[i] = v[j]; v[j] = t;
          j--;
     }
}


void load_denseLayer(denseLayer *dl, FILE * const pf, int bin)
{
     int M; fread(&M, sizeof(int), 1, pf);
     int N; fread(&N, sizeof(int), 1, pf);
     printf("dense: %d %d\n", M, N);
     ftens W = ftens_from_file(1, M, N, 1, pf);
     ftens b = ftens_from_file(1, 1, M, 1, pf);
     if (bin) ftens_sign(&W);
     denseLayer_set(&W, dl);
     ftens_free(&W);
     ftens_free(&b);
}


void load_bnormLayer(bnormLayer *bnl, FILE *pf)
{
     int N; fread(&N, sizeof(int), 1, pf);
     printf("bnorm: %d\n", N);
     ftens beta  = ftens_from_file(1, 1, N, 1, pf);
     ftens gamma = ftens_from_file(1, 1, N, 1, pf);
     ftens mean  = ftens_from_file(1, 1, N, 1, pf);
     ftens istd  = ftens_from_file(1, 1, N, 1, pf);
     bnormLayer_set(&mean, &istd, &beta, &gamma, bnl);
     ftens_free(&mean); ftens_free(&istd);
     ftens_free(&beta); ftens_free(&gamma);
}

void load_convLayer(convLayer *cl, FILE *pf, int bin, int rev)
{
     int a[7]; fread(a, sizeof(int), 7, pf);
     int p=a[0], D=a[1], M=a[2], N=a[3], L=a[4];
     printf("conv: \n");
     ftens fil = ftens_from_file(D, L, M, N, pf);
     if (bin) ftens_sign(&fil);
     if (rev)
          for (int w=0; w<D; w++)
               reverse(fil.data + (fil.MNL)*w, M*N*L);

     ftens asd = ftens_copy_tch(&fil);
     ftens b = ftens_from_file(1, 1, D, 1, pf);
     cl->Sm=a[5]; cl->Sn=a[6]; cl->p=a[0];
     convLayer_set(&asd, cl);
     ftens_free(&fil);
     ftens_free(&asd);
     ftens_free(&b);
}


void load_poolLayer(poolLayer *pl, FILE *pf)
{
     int M, N, Sm, Sn;
     printf("pool: \n");
     fread(&M,  sizeof(int), 1, pf);
     fread(&N,  sizeof(int), 1, pf);
     fread(&Sm, sizeof(int), 1, pf);
     fread(&Sn, sizeof(int), 1, pf);
     pl->M=M; pl->N=N; pl->Sm=Sm; pl->Sn=Sn;
}
