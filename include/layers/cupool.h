#ifndef ESP_CUPOOL_H
#define ESP_CUPOOL_H

#include "pool.h"
#include "cutens.h"


typedef struct {
     int M, N, Sm, Sn;
     pool_t op;
     cuftens out, mask;
} cupoolLayer;


cupoolLayer cupoolLayer_init(int M, int N, int Sm, int Sn);
void cupoolLayer_free(cupoolLayer *pl);
void cupoolLayer_convert(poolLayer *src, cupoolLayer *dst);
void cupoolLayer_forward(cuftens *t, cupoolLayer *pl);
void cupoolLayer_backward(cuftens *dt, cupoolLayer *pl);
void cupoolLayer_print(cupoolLayer *pl);


#endif /* ESP_CUPOOL_H */
