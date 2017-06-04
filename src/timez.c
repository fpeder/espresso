#include <stdio.h>
#include "timez.h"


void time_record(myTime *time)
{
     gettimeofday(time, NULL);
}


long elapsed_time(myTime start, myTime stop)
{
     long sec = stop.tv_sec - start.tv_sec;
     long usec = stop.tv_usec - start.tv_usec;
     usec += (sec * 1e6);
     printf("%ld us\t", usec);
     return usec;
}



