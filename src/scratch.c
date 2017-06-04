#include "util.h"

float *scratch;


void scratch_alloc(int len)
{
     if (len) scratch = MALLOC(float, len);
     else     scratch = NULL;
}

void scratch_free()
{
     if (scratch) {free(scratch); scratch=NULL;}
}
