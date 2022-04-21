/*
 * This file is part of clustbench project.
 *
 * @author Alexey Salnikov <salnikov@cs.msu.ru>
 *
 */

#include <mpi.h>
#include <sys/time.h>
#include <time.h>

#include "clustbench_types.h"

clustbench_time_t clustbench_get_time(void)
{
 return MPI_Wtime();
}

/*
 * Please use this function instead of MPI_Wtime or similar if
 * implementation of the MPI_Wtime works incorrect.
 */
double MYMPI_Wtime()
{
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return (double)tv.tv_sec+(double)tv.tv_usec*0.000001;
}

