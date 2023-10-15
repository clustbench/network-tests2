#ifndef __CLUSTBENCH_PLUGIN_OPERATIONS_H__
#define __CLUSTBENCH_PLUGIN_OPERATIONS_H__

#include "clustbench_types.h"
#include "benchmarks_common.h"

#ifdef __clpusplus
extern "C" {
#endif

 
typedef struct
{
    void *dynamic_library_handler;
    char *short_description;
    int (*print_help)(clustbench_benchmark_parameters_t*);
    int (*print_parameters)(clustbench_benchmark_parameters_t*);
    int (*parse_parameters)(clustbench_benchmark_parameters_t*,int,char**,int);
    int (*define_netcdf_vars)(int file_id, clustbench_benchmark_parameters_t* params);
    int (*put_netcdf_vars)(int file_id, clustbench_benchmark_parameters_t* params);
    int (*free_parameters)(clustbench_benchmark_parameters_t*);
    int (*test_function)(clustbench_time_result_t *times,  clustbench_time_t *real_times,
        int mes_length,
        int num_repeats,
        void *individual_parameters);
} clustbench_benchmark_pointers_t;

   
int clustbench_open_benchmark(const char *path_to_benchmark_code_dir, 
                             const char *benchmark_name,
                             clustbench_benchmark_pointers_t *pointers);

int clustbench_close_benchmark_lib(clustbench_benchmark_pointers_t *pointers);

int clustbench_print_list_of_benchmarks(const char *path_to_benchmark_code_dir);


#ifdef __cplusplus
}
#endif


#endif /* __CLUSTBENCH_PLUGIN_OPERATIONS_H__ */
