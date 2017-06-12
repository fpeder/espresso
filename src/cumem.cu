#include "util.cuh"
#include "cumem.h"

#define PTR_INIT(x) (x.total=0, x.used=0, x.base=NULL, x.curr=NULL)

cufmem fptr = {0, 0, NULL, NULL};
cupmem pptr = {0, 0, NULL, NULL};


int _cufmem() {return fptr.base != NULL;}
int _cupmem() {return pptr.base != NULL;}


void cufmem_alloc(int bytes)
{
     fptr.total = bytes;
     fptr.used  = 0;
     cudaMalloc(&fptr.base, bytes);
     cuASSERT(fptr.base, "err:\n");
     fptr.curr = fptr.base;
}

float *cufmem_reserve(int bytes)
{
     float *out = fptr.curr;
     cuASSERT(fptr.base, "err: cufmem not init \n");
     cuASSERT(fptr.used + bytes < fptr.total, "err: out of cufmem\n");
     fptr.used += bytes;
     fptr.curr += (bytes/sizeof(float));
     return out;
}

void cufmem_free()
{
     if (fptr.base) cudaFree(fptr.base);
     PTR_INIT(fptr);
}

void cufmem_reset()
{
     fptr.used = 0;
     fptr.curr = fptr.base;
}

void cupmem_alloc(int bytes)
{
     pptr.total = bytes;
     pptr.used  = 0;
     cudaMalloc(&pptr.base, bytes);
     cuASSERT(pptr.base, "err:\n");
     pptr.curr = pptr.base;
}

uint64_t *cupmem_reserve(int bytes)
{
     uint64_t *out = pptr.curr;
     cuASSERT(pptr.base, "err: cumem not init \n");
     cuASSERT(pptr.used + bytes < pptr.total, "err: out of cupmem\n");
     pptr.used += bytes;
     pptr.curr += (bytes/sizeof(uint64_t));
     return out;
}

void cupmem_free()
{
     if (pptr.base) cudaFree(pptr.base);
     PTR_INIT(pptr);
}

void cupmem_reset()
{
     pptr.used = 0;
     pptr.curr = pptr.base;
}
