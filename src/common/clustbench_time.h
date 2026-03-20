#ifndef __MY_TIME_H_
#define __MY_TIME_H_

#include "clustbench_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

//костыль на случаи когда основная функция из mytime.c
//не робит
extern clustbench_time_t clustbench_get_time(void);

#ifdef __cplusplus
}
#endif

#endif /* __MY_TIME_H_ */
