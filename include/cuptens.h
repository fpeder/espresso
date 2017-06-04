#ifndef ESP_CUPTENS_H
#define ESP_CUPTENS_H

#include <stdint.h>
#include "cumem.h"
#include "tens.h"
#include "cutens.h"


typedef struct {
     int D, M, N, L;
     int X, p, MNL;
     int bytes;
     uint64_t *data;
} cuptens;


cuptens cuptens_empty(int D, int M, int N, int L);
cuptens cuptens_init(int D, int M, int N, int L);
cuptens cuptens_pad(cuptens *src, int p);
cuptens cuptens_from_cupmem(int D, int M, int N, int L);
cuptens cuptens_lower_init(cuptens *src, int W, int H, int Sx, int Sy);
cuptens cuptens_lower_cupmem(cuptens *src, int W, int H, int Sx, int Sy);

cuptens cuptens_convert(ftens *t);
cuptens cuptens_convert(cuftens *t);
void cuptens_convert(cuftens *src, cuptens *dst);
void cuptens_free(cuptens *pt);
uint64_t *cuptens_dump(cuptens *t);
void cuptens_print_shape(cuptens *t);
void cuptens_print(cuptens *t);
void cuptens_print(cuptens *t, const char *fmt);
void cuptens_print_ch(cuptens *t, int w, int k, const char *fmt);

#endif /* ESP_CUPTENS_H */
