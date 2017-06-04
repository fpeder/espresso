#ifndef ESP_DENSE_H
#define ESP_DENSE_H

#include "tens.h"


#define DENSEL_INIT(dl) (                            \
          dl.M=0, dl.N=0,                            \
          dl.W .data=NULL, dl.dW .data=NULL,         \
          dl.b .data=NULL, dl.db .data=NULL,         \
          dl.in.data=NULL, dl.out.data=NULL)


#ifdef __cplusplus
extern "C" {
#endif

     typedef struct {
          int M, N;
          ftens W, b, dW, db;
          ftens in, out;
     } denseLayer;

     denseLayer denseLayer_init(int M, int N);
     void denseLayer_print_shape(denseLayer *dl);
     void denseLayer_free(denseLayer *dl);
     void denseLayer_set(ftens *W, denseLayer *dl);
     void denseLayer_forward(ftens *t, denseLayer *dl, int cpy);
     void denseLayer_backward(ftens *dt, denseLayer *dl);


#ifdef __cplusplus
}
#endif

#endif /* ESP_DENSE_H */
