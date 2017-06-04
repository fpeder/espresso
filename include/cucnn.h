#ifndef ESP_CUMLP_H
#define ESP_CUMLP_H

#include "culayers.h"
#include "cnn.h"


typedef struct {
     int Ncl, Npl, Ndl, Nbnl;
     cuinputLayer il;
     cuconvLayer  *cl;
     cupoolLayer  *pl;
     cudenseLayer *dl;
     cubnormLayer *bnl;
} cucnn;


cucnn cucnn_init(int Ncl, int Npl, int Ndl, int Nbnl);
void cucnn_print(cucnn *net);
void cucnn_free(cucnn *net);
cucnn cucnn_convert(cnn *net);


#endif /* ESP_CUMLP_H */
