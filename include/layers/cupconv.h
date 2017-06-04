#ifndef ESP_CUPCONV_H
#define ESP_CUPCONV_H

#include "conv.h"
#include "cuptens.h"


#define CUPCONVL_INIT(cl)                                              \
     (cl.D=0,  cl.M=0,  cl.N=0, cl.L=0,                                \
      cl.Sm=0, cl.Sn=0, cl.p=0,                                        \
      cl.W.  data=NULL, cl.pout.data=NULL,                             \
      cl.dW. data=NULL, cl.out .data=NULL,                             \
      cl.fix.data=NULL, cl.bfix.data=NULL,                             \
      cl.in. data=NULL)


typedef struct {
     int D, M, N, L, Sm, Sn, p;
     cuptens W, pout;
     cuftens dW, out, in;
     cuftens fix, bfix;
} cupconvLayer;


cupconvLayer cupconvLayer_init();
cupconvLayer cupconvLayer_init(int Sm, int Sn, int p);
void cupconvLayer_convert(convLayer *src, cupconvLayer *dst, int fix);
void cupconvLayer_set(ftens *W, cupconvLayer *cl, int fix);
void cupconvLayer_forward(cuptens *t, cupconvLayer *cl);
void cupconvLayer_forward_initial(cuftens *t, cupconvLayer *cl, float norm);
void cupconvLayer_free(cupconvLayer *cl);



#endif /* ESP_CUPCONV_H */
