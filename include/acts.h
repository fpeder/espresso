#ifndef ESP_ACTIVATIONS_H
#define ESP_ACTIVATIONS_H

#include "tens.h"

#ifdef __cplusplus
extern "C" {
#endif

     void signAct_forward(ftens *t);
     void signAct_backward(ftens *dout);
     void reluAct_forward(ftens *t);
     void reluAct_backward(ftens *dout);
     void softmaxAct_forward(ftens *t);
     void softmaxAct_backward(ftens *dout);


#ifdef __cplusplus
}
#endif


#endif /* ESP_ACTIVATIONS_H */
