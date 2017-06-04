#ifndef ESP_TIME_H
#define ESP_TIME_H

#include <sys/time.h>

#define TIME_START()       time_record(&aaat1)
#define TIME_STOP()        time_record(&aaat2) 
#define TIME_STOP_PRINT()  TIME_STOP(); elapsed_time(aaat1, aaat2)
#define TIME_STOP_SAVE(t)  TIME_STOP(); t = elapsed_time(aaat1, aaat2)


typedef struct timeval myTime;
myTime aaat1, aaat2;

void time_record(myTime *time);
long elapsed_time(myTime start, myTime stop);

#endif /* ESP_TIME_H */



