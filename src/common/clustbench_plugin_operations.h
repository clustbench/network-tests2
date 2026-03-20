#ifndef __CLUSTBENCH_PLUGIN_OPERATIONS_H__
#define __CLUSTBENCH_PLUGIN_OPERATIONS_H__

#include "clustbench_types.h"
#include "delay_measurements_amount_auto.h"
#include "benchmarks_common.h"

#ifdef __clpusplus
extern "C" {
#endif

 
typedef struct
{
    void *dynamic_library_handler;
    char *short_description;
    //что это такое?
    int (*print_help)(clustbench_benchmark_parameters_t*);
    int (*print_parameters)(clustbench_benchmark_parameters_t*);
    int (*parse_parameters)(clustbench_benchmark_parameters_t*,int,char**,int);
    int (*define_netcdf_vars)(int file_id, clustbench_benchmark_parameters_t* params);
    int (*put_netcdf_vars)(int file_id, clustbench_benchmark_parameters_t* params);
    int (*free_parameters)(clustbench_benchmark_parameters_t*);
    // If timer type = 0, default timer is used. Else CLOCK_REALTIME is used.
    int (*test_function)(clustbench_time_result_t *times,  clustbench_time_t *real_times,
        int mes_length,
        int num_repeats,
        int timer_type,
        struct AlgorithmMainInfo * algo_main_info
    );
    // If timer type = 0, default timer is used. Else CLOCK_REALTIME is used.
    int (*test_function_mashed)(clustbench_time_result_t **times, clustbench_time_t **real_times,
                                int num_repeats,
                                int beg_length,
                                int step_length,
                                int end_length,
                                int permutation_length,
                                int amount_of_lengths,
                                int *permutation,
                                int timer_type
    );
} clustbench_benchmark_pointers_t;

   
int clustbench_open_benchmark(const char *path_to_benchmark_code_dir, 
                             const char *benchmark_name,
                             clustbench_benchmark_pointers_t *pointers);

int clustbench_close_benchmark_lib(clustbench_benchmark_pointers_t *pointers);

int clustbench_print_list_of_benchmarks(const char *path_to_benchmark_code_dir);

/// @brief Функция, вычисляющая рассинхрон процесса с процессом с рангом 0.
/// @param rank Ранг текущего процесса.
/// @param commSize Количество процессов в коммуникаторе.
/// @return Значение рассинхрона (в наносекундах).
int calculate_offsets(int rank, int commSize);

/// @brief Функция, синхронизирующая процессы.
/// @param rank Ранг текущего процесса.
/// @param commSize Количество процессов в коммуникаторе.
/// @param offset Рассинхрон (в обвчном режиме равен нулю).
void sync_time(int rank, int commSize, long offset);

#ifdef __cplusplus
}
#endif


#endif /* __CLUSTBENCH_PLUGIN_OPERATIONS_H__ */
