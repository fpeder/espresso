#ifndef ESP_CUACT_H
#define ESP_CUACT_H

#include "cuptens.h"


void cusignAct_forward(cuftens *t);
void cusignAct_backward(cuftens *dout);
void cupsignAct_forward(cuftens *src, cuptens *out);



#endif /* ESP_CUACT_H */
