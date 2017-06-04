#include <stdint.h>
#include <string.h>

#include "util.h"
#include "tens.h"

#define W 28
#define H 28
#define C 1
#define IMG_B W*H
#define LAB_B 1
#define TRAIN_IMG 6000
#define TRAIN_HB 16
#define TEST_HB 8


void mnist_load_X(const char *tf, int start, int num,
                      ftens *X)
{
     ASSERT(start + num < TRAIN_IMG, "mnist err img");
     uint8_t buff[W*H];
     FILE *pf = fopen(tf, "rb"); ASSERT(pf, "mnist err tf");
     fseek(pf, TRAIN_HB + start * IMG_B, SEEK_SET);
     for (int i=0; i < num; i++) {
          float *out = X->data + i*X->MNL;
          FREAD(buff, uint8_t, IMG_B, pf, "mnist err read");
          for (int i=0; i < W*H; i++)
               out[i] = (float) buff[i];
     }
     fclose(pf);
}


void mnist_load_y(const char *lf, int start, int num, ftens *y)
{
     ASSERT(start + num < TRAIN_IMG, "mnist err lab\n");
     float *out = y->data; uint8_t buff;
     FILE *pf = fopen(lf, "rb"); ASSERT(pf, "mnist err rl\n");
     fseek(pf, TEST_HB + start * LAB_B, SEEK_SET);
     for (int i=0; i < num; i++) {
          memset(out, 0, sizeof(float) * 10);
          fread(&buff, sizeof(uint8_t), 1, pf);
          out[buff] = 1.0f;
          out += y->MNL;
     }
}


#undef W
#undef H
#undef C
#undef HEADER_BYTES
#undef IMG_B
#undef LAB_B
#undef TRAIN_IMG
#undef TRAIN_HB
#undef TEST_HB
