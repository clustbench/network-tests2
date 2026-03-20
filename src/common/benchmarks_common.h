/*
 *  This file is a part of the ClustBench project.
 *  Copyright (C) 2020  Alexey N. Salnikov
 *
 *  File contains some declarions using  by
 *  benchmarks and other tools in this project.
 *
 *  Authors:
 *      Alexey N. Salnikov (salnikov@cs.msu.ru)
 *
 */

#ifndef __BENCHMARKS_COMMON_H__
#define __BENCHMARKS_COMMON_H__

#include "clustbench_types.h"


typedef struct
{
    clustbench_time_t average;
    clustbench_time_t median;
    clustbench_time_t deviation;
    clustbench_time_t min;
    int amount_of_measurements;
} clustbench_time_result_t;

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * The function comparator. It is used for
     * array of delays sorting at the moment benchmark is counting a mediane.
     */
    int clustbench_time_cmp(const void *a, const void *b);

    /**
     * This function Buld a file contains names of cluster nodes in
     * order corresponding to the rank of mpi_process.
     */
    int clustbench_create_hosts_file
    (
    	 const clustbench_benchmark_parameters_t *parameters,
    	 char **hosts_names
    );

#ifdef __cplusplus
}
#endif

/*
 * The send process is an analog of i position
 * in marix when all coords counts in forward oder by rows
 * (0,0)->0
 * (0,1)->1
 * ...
 * (1,0)-> size
 * (1,1)-> size+1
 * ...
 * (size-1,size-1) size*size-1
 * */
#define GET_SEND_PROCESS(squere_coord,size) (squere_coord)/(size)
/*
 * The recv process is an analog of j position
 * in marix when all coords counts in forward oder by rows
 * (0,0)->0
 * (0,1)->1
 * ...
 * (1,0)-> size
 * (1,1)-> size+1
 * ...
 * (size-1,size-1) size*size-1
 * */
#define GET_RECV_PROSESS(squere_coord,size) (squere_coord)%(size)



#endif /* __BENCHMARKS_COMMON_H__  */


