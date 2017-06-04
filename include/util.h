#ifndef ESP_UTIL_H
#define ESP_UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "scratch.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ID2(i, j, N)        ((i)*(N) + (j))
#define ID3(i, j, k, N, L)  (ID2(i,j,N)*(L) + (k))
#define IDX(i, j, k, N, L)  (((i)*N + (j))*(L) + (k))

#define SET(x, y)           (x).y = y
#define PSET(x, y)          (x)->y = y

#define PAD(x, p)           ((x) + ((p)<<1))
#define CEIL(x, y)          (((x) + (y) - 1) / (y))
#define ROUND_UP(x, y)      (CEIL(x, y) * (y))
#define MAX(x, y)           ((x) > (y) ? (x) : (y))
#define MIN(x, y)           ((x) < (y) ? (x) : (y))
#define OUT_LEN(x, y, z)    (ceilf(((x)-(y)+1)/(float)z))

#define SP                  printf(" ");
#define NL                  printf("\n");
#define SEP                 printf(" | ");
#define D(t)                (t.data)
#define LEN(t)              (t.bytes/sizeof(float))
#define FOR(x,y,n)          for (int x=y; x<(n); x++)
#define READ_INT(x, pf)     FREAD(x, sizeof(int), 1, pf, "asd")
#define READ_UINT8(x, pf)   FREAD(x, sizeof(uint8_t), 1, pf, "asd")
#define MALLOC(type, num)   (type *) malloc((num) * sizeof(type))
#define BYTES(type, num)    ((num) * sizeof(type))
#define CALLOC(type, num)   (type *) calloc(num, sizeof(type))

#define FREAD(x, type, num, pf, msg)                        \
     if (fread(x, sizeof(type), num, pf) != num) {          \
          fprintf(stderr, msg);                             \
          exit(2);                                         \
     }

#ifdef NDEBUG
#define ASSERT(exp, msg) assert(EXP && MSG)

#else
#define ASSERT(exp, msg)                         \
     if(!(exp)) {                                \
          fprintf(stderr, msg);                  \
          exit(1);                               \
     }
#endif


#ifdef __cplusplus
}
#endif

#endif /* ESP_UTIL_H */
