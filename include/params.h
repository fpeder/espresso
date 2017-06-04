#ifndef ESP_PARAMS_H
#define ESP_PARAMS_H

#include "layers.h"

#define INPUTL  0
#define CONVL   3
#define POOLL   4
#define DENSEL  1
#define BNORML  2
#define LNUM    (1<<4)
#define LDAT    (2<<4)

#ifdef __cplusplus
extern "C" {
#endif

     void load_denseLayer(denseLayer *dl, FILE * const pf, int bin);
     void load_bnormLayer(bnormLayer *bnl, FILE * const pf);
     void load_convLayer(convLayer *cl, FILE * const pf,
                         int bin, int rev);
     void load_poolLayer(poolLayer *pl, FILE * const pf);


#ifdef __cplusplus
}
#endif


#endif /* ESP_PARAMS_H */
