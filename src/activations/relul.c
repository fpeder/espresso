#include "tens.h"
#include "util.h"


void relu_forward(ftens *t)
{
     const int len = ftens_len(t);
     for (int i=0; i < len; i++)
          t->data[i] = MAX(0, t->data[i]);
}

void relu_backward(ftens *dout)
{
     fprintf(stderr, "not implemeted yer\n");
     exit(-4);
}
