#ifndef ESP_CUCONV_H
#define ESP_CUCONV_H

#include "cutens.h"
#include "conv.h"


typedef struct {
     int D, M, N, L, Sm, Sn, p;
     cuftens W, b, out;
     cuftens dW, db, in;
} cuconvLayer;


cuconvLayer cuconvLayer_init(int Sm, int Sn, int p);
void cuconvLayer_free(cuconvLayer *cl);
void cuftens_print(cuftens *t);
void cuconvLayer_print_shape(cuconvLayer *cl);
void cuconvLayer_convert(convLayer *src, cuconvLayer *dst);
void cuconvLayer_set(ftens *W, cuconvLayer *cl);
cuftens cuconvLayer_pad_input(cuftens *t, int p);
cuftens cuconvLayer_lower_input(cuftens *t, cuconvLayer *cl);
void cuconvLayer_forward(cuftens *t, cuconvLayer *cl, int save);
void cuconvLayer_backward(cuftens *dout, cuconvLayer *cl);


#endif /* ESP_CUCONV_H */
