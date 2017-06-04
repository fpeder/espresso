#include <stdio.h>
#include <stdlib.h>

#include "util.h"

#define FPTR d_fscratch
#define PPTR d_pscratch
#define FPTR_INC(x, len) d_fscratch ? x + len : NULL
#define PPTR_INC(x, len) d_pscratch ? x + len : NULL

#define cuHtoD    cudaMemcpyHostToDevice
#define cuDtoH    cudaMemcpyDeviceToHost
#define cuDtoD    cudaMemcpyDeviceToDevice

#ifndef cuD
#define cuD(x) ((x).d_data)
#endif

#define cuASSERT(exp, msg)                       \
  if (!(exp)) {                                  \
       fprintf(stderr, msg);                     \
       exit(-1);                                 \
  }

#define CUDA_SAFE_CALL(call)                                          \
{                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}


#define CHECK_LAUNCH_ERROR()                                          \
{                                                                  \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}
