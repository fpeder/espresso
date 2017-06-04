#ifndef ESP_BNORM_H
#define ESP_BNORM_H

#include "tens.h"

#define BNORML_INIT(bnl) {                                       \
          bnl.N=0; bnl.ug=0;                                    \
          bnl.mean. data=NULL; bnl.istd  .data=NULL;            \
          bnl.gmean.data=NULL; bnl.gistd .data=NULL;            \
          bnl.beta .data=NULL; bnl.gamma .data=NULL;            \
          bnl.dbeta.data=NULL; bnl.dgamma.data=NULL;            \
          bnl.in.   data=NULL; bnl.tmp.   data=NULL;            \
     }


#ifdef __cplusplus
extern "C" {
#endif

     typedef struct {
          int N, ug;
          ftens mean,  istd, gmean,  gistd;
          ftens gamma, beta, dgamma, dbeta;
          ftens in, tmp;
     } bnormLayer;


     bnormLayer bnormLayer_init(int use_global);
     void bnormLayer_free(bnormLayer *bnl);
     void bnormLayer_forward(ftens *t, bnormLayer *bnl, int save);
     void bnormLayer_backward(ftens *dt, bnormLayer *bnl);
     void bnormLayer_update(bnormLayer *bnl);
     void bnormLayer_set(ftens *mean,  ftens *istd,
                         ftens *gamma, ftens *beta, bnormLayer *bnl);


#ifdef __cplusplus
}
#endif

#endif /* ESP_BNORM_H */
