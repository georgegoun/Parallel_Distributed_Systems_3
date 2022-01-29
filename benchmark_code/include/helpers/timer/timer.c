#include "timer.h"

#include <stdio.h>
#include <time.h>

struct timespec timerStart(struct timespec start)
{
    clock_gettime(CLOCK_MONOTONIC, &start);
    return start;
}
struct timespec timerStop(struct timespec stop)
{
    clock_gettime(CLOCK_MONOTONIC, &stop);
    return stop;
}
double timeDif(struct timespec start_, struct timespec stop_)
{
    double time_dif;
    time_dif = (stop_.tv_sec - start_.tv_sec) * 1e9;
    time_dif = (time_dif + (stop_.tv_nsec - start_.tv_nsec)) * 1e-9;
    return time_dif;
}