#ifndef ESP_POOL_H
#define ESP_POOL_H

#include "tens.h"

#define POOLL_INIT(pl)                                  \
     (pl.M=0, pl.N=0, pl.Sm=0, pl.Sn=0, pl.op=MAX,      \
      pl.out.data=NULL, pl.mask.data=NULL)


#ifdef __cplusplus
extern "C" {
#endif

     typedef enum {MAX, AVG} pool_t;

     typedef struct {
          int M, N, Sm, Sn; pool_t op;
          ftens out, mask;
     } poolLayer;


     poolLayer poolLayer_init(int M, int N, int Sm, int Sn);
     void poolLayer_free(poolLayer *pl);
     void poolLayer_forward(ftens *t, poolLayer *pl);
     void poolLayer_backward(ftens *dout, poolLayer *pl);


#ifdef __cplusplus
}
#endif


#endif /* ESP_POOL_H */
