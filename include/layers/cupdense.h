#ifndef ESP_CUPDENSE_H
#define ESP_CUPDENSE_H

#include "cuptens.h"
#include "dense.h"


#define CUPDENSEL_INIT(dl) \
     (dl.M=0, dl.N=0,                                              \
      dl.W. data=NULL, dl.in. data=NULL, dl.pout.data=NULL,        \
      dl.dW.data=NULL, dl.out.data=NULL, dl.fix.data=NULL)


typedef struct {
     int M, N;
     cuptens W, in, pout;
     cuftens dW, out, fix;
} cupdenseLayer;


cupdenseLayer cupdenseLayer_init();
void cupdenseLayer_convert(denseLayer *src, cupdenseLayer *dst, int fix);
void cupdenseLayer_free(cupdenseLayer *dl);
void cupdenseLayer_print_size(cupdenseLayer *dl);
void cupdenseLayer_set(ftens *W, cupdenseLayer *dl, int fix);
void cupdenseLayer_forward(cuptens *t, cupdenseLayer *dl, int save);
void cupdenseLayer_forward_initial(cuftens *t, cupdenseLayer *dl, float norm);


#endif /* ESP_CUPDENSE_H */
