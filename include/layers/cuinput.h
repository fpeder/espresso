#ifndef ESP_CUINPUT_H
#define ESP_CUINPUT_H

#include "tens.h"
#include "cutens.h"


typedef struct {
     cuftens out;
} cuinputLayer;


void cuinputLayer_forward(ftens *t, cuinputLayer *il, int norm);
void cuinputLayer_free(cuinputLayer *il);


#endif /* ESP_CUINPUT_H */
