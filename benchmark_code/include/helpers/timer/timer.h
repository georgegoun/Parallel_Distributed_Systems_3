#ifndef HEADER_FILE_TIME
#define HEADER_FILE_TIME

#include <stdint.h>
#include <stdio.h>

struct timespec timerStart(struct timespec start);
struct timespec timerStop(struct timespec stop);

double timeDif(struct timespec start_, struct timespec stop_);
#endif