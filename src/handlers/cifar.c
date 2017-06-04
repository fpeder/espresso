#include "util.h"
#include "tens.h"

#define W 32
#define H 32
#define L 3
#define TRAIN_IMG 50000
#define TEST_IMG  10000


void cifar10_load_Xy(const char *tf, int start, int num,
                     ftens *X, ftens *y)
{
     ASSERT(start + num < TEST_IMG, "err: cifar num\n");
     ASSERT(X->MNL == W*H*L, "err: input shape\n");
     uint8_t X_buff[W*H*L];
     uint8_t y_buff;
     FILE *pf = fopen(tf, "rb");
     ASSERT(pf, "err: fopen \n");
     ftens tmpX = ftens_init(num, L, W, H);
     ftens_clear(y);
     fseek(pf, (W*H*L+1)*start, SEEK_SET);
     for (int i=0; i < num; i++) {
          float *outX = tmpX.data + i * tmpX.MNL;
          float *outy = y->data   + i * y->MNL;
          fread(&y_buff, sizeof(uint8_t), 1,     pf);
          fread(X_buff,  sizeof(uint8_t), W*H*L, pf);
          outy[y_buff] = 1.0f;
          for (int i=0; i < W*H*L; i++)
               outX[i] = (float) X_buff[i];
     }

     ftens_tch(&tmpX, X);
     ftens_free(&tmpX);
}

#undef W
#undef H
#undef L
