#ifndef ESP_CUBNORN_H
#define ESP_CUBNORN_H

#include "cutens.h"
#include "bnorm.h"


typedef struct {
     int N, ug;
     cuftens mean,  istd, gmean,  gistd;
     cuftens gamma, beta, dgamma, dbeta;
     cuftens in, tmp;
} cubnormLayer;


cubnormLayer cubnormLayer_init(int use_global);
void cubnormLayer_convert(bnormLayer *src, cubnormLayer *dst);
void cubnormLayer_free(cubnormLayer *bnl);
void cubnormLayer_forward(cuftens *t, cubnormLayer *bnl, int save);
void cubnormLayer_backward(cuftens *dt, cubnormLayer *bnl);
void cubnormLayer_update(cubnormLayer *bnl);
void cubnormLayer_set(ftens *mean,  ftens *istd,
                      ftens *gamma, ftens *beta,
                      cubnormLayer *bnl);


#endif /* ESP_CUBNORN_H */
