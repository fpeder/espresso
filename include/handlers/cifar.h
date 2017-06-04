#ifndef ESP_CIFAR_H
#define ESP_CIFAR_H

#ifdef __cplusplus
extern "C" {
#endif

     void cifar10_load_Xy(const char *tf, int start, int num,
                          ftens *X, ftens *y);

#ifdef __cplusplus
}
#endif

#endif /* ESP_CIFAR_H */
