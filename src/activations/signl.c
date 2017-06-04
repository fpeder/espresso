#include "util.h"
#include "tens.h"

void signAct_forward(ftens *t)
{
     ftens_sign(t);
     //const int len = ftens_len(t);
     //for (int i=0; i < len; i++)
     //t->data[i] = 2.0f * (t->data[i] > 0.0f) - 1.0f;
}


void signAct_backward(ftens *t)
{
     fprintf(stderr, "not implemeted yet\n");
     exit(-4);
}
