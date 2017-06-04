#include "util.h"
#include "input.h"


void inputLayer_load(ftens *t, inputLayer *il)
{
     il->out = ftens_copy(t);
}


void inputLayer_free(inputLayer *il)
{
     ftens_free(&il->out);
}


void inputLayer_forward(inputLayer *il)
{
     if (!il->out.data) {
          fprintf(stderr, "err: in null\n");
          exit(-1);
     }

     float *ptr = il->out.data;
     const int len = ftens_len(&il->out);
     for (int i=0; i < len; i++)
          ptr[i] = 2.0f * ptr[i]/255.0f - 1.0f;
}


/* void inputLayer_pad(inputLayer *il, const int p) */
/* { */
/*      ftens tmp = ftens_copy_pad(&il->out, p); */
/*      ftens_free(&il->out); */
/*      il->out = tmp; */
/* } */
