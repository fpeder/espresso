#ifndef ESP_CUMLP_H
#define ESP_CUMLP_H

#include "mlp.h"
#include "culayers.h"

typedef struct {
     int Ndl, Nbnl;
     cuinputLayer il;
     cudenseLayer *dl;
     cubnormLayer *bnl;
} cumlp;


cumlp cumlp_init(int Ndl, int Nbnl);
cumlp cumlp_convert(mlp *net);

void cumlp_print(cumlp *net);
void cumlp_free(cumlp *net);



#endif /* ESP_CUMLP_H */
