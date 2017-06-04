#ifndef ESP_CNN_H
#define ESP_CNN_H

#include "tens.h"
#include "layers.h"


#ifdef __cplusplus
extern "C" {
#endif

     typedef struct {
          int Ncl, Npl, Ndl, Nbnl;
          inputLayer il;
          convLayer  *cl;
          poolLayer  *pl;
          denseLayer *dl;
          bnormLayer *bnl;
     } cnn;

     cnn cnn_init(int Ncl, int Npl, int Ndl, int Nbnl);
     cnn cnn_load(const char *esp, int bin, int rev);
     void cnn_free(cnn *net);
     void cnn_print(cnn *net);


#ifdef __cplusplus
}
#endif

#endif /* ESP_CNN_H */
