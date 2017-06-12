#ifndef ESP_MLP_H
#define ESP_MLP_H

#include "layers.h"


#ifdef __cplusplus
extern "C" {
#endif

     typedef struct {
          int Ndl, Nbnl;
          inputLayer il;
          denseLayer *dl;
          bnormLayer *bnl;
     } mlp;


     mlp  mlp_init(int Ndl, int Nbnl);
     mlp  mlp_load(const char *esp, int bin);
     void mlp_free(mlp *net);
     void mlp_print(mlp *net);


#ifdef __cplusplus
}
#endif


#endif /* ESP_MLP_H */
