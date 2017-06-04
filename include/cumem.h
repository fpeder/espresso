#ifndef ESP_CUMEM_H
#define ESP_CUMEM_H

#include <stdint.h>

#define CUFMEM  _cufmem()
#define CUPMEM  _cupmem()


typedef struct {
     int total, used;
     float *base;
     float *curr;
} cufmem;


typedef struct {
     int total, used;
     uint64_t *base;
     uint64_t *curr;
} cupmem;


int _cufmem();
int _cupmem();

void cufmem_alloc(int len);
float *cufmem_reserve(int len);
void cufmem_free();
void cufmem_reset();

void cupmem_alloc(int len);
uint64_t *cupmem_reserve(int len);
void cupmem_free();
void cupmem_reset();


#endif /* ESP_CUMEM_H */
