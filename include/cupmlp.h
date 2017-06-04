#ifndef ESP_PMLP_H
#define ESP_PMLP_H

#include "mlp.h"
#include "culayers.h"


typedef struct {
     int Ndl, Nbnl;
     cuinputLayer il;
     cupdenseLayer *dl;
     cubnormLayer *bnl;
} cupmlp;


cupmlp cupmlp_init(int Ndl, int Nbnl);
cupmlp cupmlp_convert(mlp *nn);

void cupmlp_print(cupmlp *nn);
void cupmlp_free(cupmlp *nn);



#endif /* ESP_PMLP_H */
