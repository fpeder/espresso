#ifndef ESP_CUDENSE_H
#define ESP_CUDENSE_H

#include "cutens.h"
#include "dense.h"


typedef struct {
     int M, N;
     cuftens W, b, out;
     cuftens dW, db, in;
} cudenseLayer;


cudenseLayer cudenseLayer_init(int M, int N);
void cudenseLayer_convert(denseLayer *src, cudenseLayer *dst);
void cudenseLayer_free(cudenseLayer *dl);
void cudenseLayer_print_size(cudenseLayer *dl);
void cudenseLayer_set(ftens *W, cudenseLayer *dl);
void cudenseLayer_forward(cuftens *t, cudenseLayer *dl, int save);
void cudenseLayer_backward(cuftens *dt, cudenseLayer *dl);


#endif /* ESP_CUDENSE_H */
