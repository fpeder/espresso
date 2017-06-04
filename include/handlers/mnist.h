#ifndef ESP_MNIST_H
#define ESP_MNIST_H

#include "tens.h"

#ifdef __cplusplus
extern "C" {
#endif


void mnist_load_X(const char *tf, int start, int num,
                  ftens *X);

void  mnist_load_y(const char *lf, int start, int num,
                   ftens *y);


#ifdef __cplusplus
}
#endif

#endif /* ESP_MNIST_H */
