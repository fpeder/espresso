#include "norm.cuh"
#include "cutens.h"


void cunorm (cuftens *t)
{
     const int BS=32;
     const int len = t->bytes/sizeof(float);

     ker_norm <<<CEIL(len, BS), BS>>> (t->data, len);
}
