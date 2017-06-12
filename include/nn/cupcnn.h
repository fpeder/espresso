#ifndef ESP_CUPCNN_H
#define ESP_CUPCNN_H

#include "culayers.h"
#include "cnn.h"


typedef struct {
     int Ncl, Npl, Ndl, Nbnl;
     cuinputLayer  il;
     cupconvLayer  *cl;
     cupoolLayer   *pl;
     cupdenseLayer *dl;
     cubnormLayer  *bnl;
} cupcnn;


cupcnn cupcnn_init(int Ncl, int Npl, int Ndl, int Nbnl);
cupcnn cupcnn_convert(cnn *nn);
void cupcnn_print(cupcnn *nn);
void cupcnn_free(cupcnn *nn);


#endif /* ESP_CUPCNN_H */
