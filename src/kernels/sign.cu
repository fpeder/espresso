#include "util.cuh"
#include "cutens.h"
#include "sign.cuh"


void cusign(cuftens *a)
{
     const int BS=32;
     const int len = cuftens_len(a);
     ker_sign <<<CEIL(len, BS), BS>>> (a->data, len);
}
