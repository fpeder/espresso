#ifndef ESP_CONV_H
#define ESP_CONV_H

#include "tens.h"

#define CONVL_INIT(cl) {                                       \
          cl.D=0; cl.M=0; cl.N=0; cl.L=0;                      \
          cl.Sm=0; cl.Sn=0; cl.p=0;                            \
          cl.W.data=NULL;  cl.b.data=NULL; cl.out.data=NULL;   \
          cl.dW.data=NULL; cl.db.data=NULL; cl.in.data=NULL;   \
}

#ifdef __cplusplus
extern "C" {
#endif

     typedef struct {
          int D, M, N, L, Sm, Sn, p;
          ftens W, b, out;
          ftens dW, db, in;
     } convLayer;

     convLayer convLayer_init(int Sm, int Sn, int p);
     void convLayer_print_shape(convLayer *cl);
     void convLayer_free(convLayer *cl);
     void convLayer_set(ftens *W, convLayer *cl);
     void convLayer_forward(ftens *t, convLayer *cl, int save);

#ifdef __cplusplus
}
#endif

#endif /* ESP_CONV_H */
