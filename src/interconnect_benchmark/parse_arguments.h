#ifndef __CLUSTBENCH_PARSE_ARGUMENTS_H__
#define __CLUSTBENCH_PARSE_ARGUMENTS_H__

#include "clustbench_types.h"

#define VERSION_FLAG 1
#define ERROR_FLAG   -1
#define HELP_FLAG    2
#define LIST_FLAG    3

#define UNKNOWN_FLAG 3

#ifdef __cplusplus
extern "C"
{
#endif

    int print_benchmark_help_message (clustbench_benchmark_parameters_t *parameters);

    int print_network_test_help_message(clustbench_benchmark_parameters_t *parameters);

    int parse_network_test_arguments(clustbench_benchmark_parameters_t *parameters,
                                     int argc, 
                                     char **argv,
                                     int mpi_rank
                                    );


    int print_network_test_parameters(clustbench_benchmark_parameters_t *parameters);

    int print_individual_benchmark_parameters(clustbench_benchmark_parameters_t *parameters);

    int parse_individual_benchmark_parameters(clustbench_benchmark_parameters_t *parameters,
                                          int argc, char **argv,int mpi_rank);
                                          
#ifdef __cplusplus
}
#endif

#endif /* __CLUSTBENCH_PARSE_ARGUMENTS_H__ */

