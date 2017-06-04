#ifndef ESP_TENS_H
#define ESP_TENS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

     typedef struct {
          int D, M, N, L, MNL;
          int bytes;
          float *data;
     } ftens;


     ftens ftens_init(int D, int M, int N, int L);
     ftens ftens_zeros(int D, int M, int N, int L);
     ftens ftens_ones(int D, int M, int N, int L);
     ftens ftens_rand(int D, int M, int N, int L);
     ftens ftens_rand_range(int D, int M, int N, int L,
                            float min, float max);

     ftens ftens_copy(ftens *in);
     ftens ftens_copy_pad(ftens *t, int p);

     ftens ftens_from_ptr(int D, int M, int N, int L, float *ptr);
     ftens ftens_from_file(int D, int M, int N, int L, FILE *pf);

     ftens ftens_copy_tch(ftens *a);
     void  ftens_tch(ftens *a, ftens *b);
     void  ftens_clear(ftens *t);
     void  ftens_reshape(ftens *t, int D, int M, int N, int L);
     void  ftens_pad(ftens *src, ftens *dst, int p);
     void  ftens_maxpool(ftens *src, ftens *dst, int W, int H,
                         int Sx, int Sy);

     void ftens_lower(ftens *src, ftens *dst,
                      int W, int H, int Sx, int Sy);

     void ftens_sign(ftens *t);
     void ftens_free(ftens *t);
     void ftens_print_shape(ftens *t);
     void ftens_print(ftens *t, const char *fmt);
     void ftens_print_ch(ftens *t, int w, int k, int I, int J,
                         const char *fmt);

     static inline
     int ftens_len(ftens *t) {return t->bytes/sizeof(float);}


#ifdef __cplusplus
}
#endif

#endif /* ESP_TENS_H */
