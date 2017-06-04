#ifndef ESP_INPUT_H
#define ESP_INPUT_H

#include "tens.h"

#ifdef __cplusplus
extern "C" {
#endif

     typedef struct {
          ftens out;
     } inputLayer;

     void inputLayer_load(ftens *in, inputLayer *il);
     void inputLayer_free(inputLayer *il);
     void inputLayer_forward(inputLayer *il);
     void inputLayer_pad(inputLayer *il, int p);


#ifdef __cplusplus
}
#endif

#endif /* ESP_INPUT_H */
