#include <stdio.h>
#include "kernels.h"


void cusignAct_forward(cuftens *t)
{
     cusign(t);
}


void cusignAct_backward(cuftens *dout)
{
     fprintf(stderr, "not implemented yet\n");
     exit(-4);
}
