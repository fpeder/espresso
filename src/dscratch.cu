#include "util.cuh"

float    *d_fscratch;
uint64_t *d_pscratch;


void dfscratch_alloc(int len)
{
     if (len) cudaMalloc(&d_fscratch, BYTES(float, len));
     else     d_fscratch = NULL;
}

void dfscratch_free()
{
     if (d_fscratch) {cudaFree(d_fscratch); d_fscratch=NULL;}
}

void  dpscratch_alloc(int len)
{
     if (len) cudaMalloc(&d_pscratch, BYTES(uint64_t, len));
     else     d_pscratch = NULL;
}

void dpscratch_free()
{
     if (d_pscratch) {cudaFree(d_pscratch); d_pscratch=NULL;}
}
