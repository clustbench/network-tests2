#ifndef __CLUSTBENCH_TYPES_H__
#define __CLUSTBENCH_TYPES_H__

#define CLUSTBENCH_MPI_TIME_T MPI_DOUBLE

#define CLUSTBENCH_HOSTNAME_LENGTH 1024

#define CLUSTBENCH_MIN             1
#define CLUSTBENCH_DEVIATION       2
#define CLUSTBENCH_AVERAGE         4
#define CLUSTBENCH_MEDIAN          8
#define CLUSTBENCH_ALL     16

typedef double clustbench_time_t;
typedef double *clustbench_array_time_t;

typedef struct
{
    int  num_procs;
    const char *benchmark_name;
    unsigned int  begin_message_length;
    unsigned int  end_message_length;
    unsigned int  step_length;
    unsigned int  num_repeats;
    unsigned int statistics_save; /* bitmask */
    const char *file_name_prefix;
    const char *path_to_benchmark_code_dir;
    void *benchmark_parameters;

} clustbench_benchmark_parameters_t;

#endif /*__CLUSTBENCH_TYPES_H__ */

