#include "my_time.h"
#include "my_malloc.h"
#include "tests_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

extern int comm_rank;
extern int comm_size;

int one_to_one_cuda( Test_time_result_type * times, int mes_length, int num_repeats );

int one_to_one_cuda( Test_time_result_type * times, int mes_length, int num_repeats )
{
    return 0;
}