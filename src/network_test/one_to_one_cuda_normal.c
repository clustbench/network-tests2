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
extern int* gpu_count;
extern int total_gpu;

Test_time_result_type calc_stats( px_my_time_type* all_times, int num_repeats );

void real_one_to_one_cuda( Test_time_result_type *times, int mes_length, int num_repeats, int source_proc, int dest_proc,
                           int source_gpu, int dest_gpu );

int one_to_one_cuda( Test_time_result_type * times, int mes_length, int num_repeats )
{

    int i, j, k, n, m;

    int gpu_sr[4];

    int conf = 1;
    int send_proc, recv_proc;
    int send_gpu, recv_gpu;
    MPI_Status status;

    if ( comm_rank == 0 )
    {
        for ( n = 0; n < comm_size; n++ )
        {
            for ( m = 0; m < comm_size; m++ )
            {
                send_proc = n;
                recv_proc = m;

                gpu_sr[0] = send_proc;
                gpu_sr[2] = recv_proc;
    //!TODO: add adequate treat to 0 comm_rank
                if ( send_proc == 0 || recv_proc == 0)
                {
                    continue;
                }

                for ( j = 0; j < gpu_count[send_proc]; j++ )
                    for ( k = 0; k < gpu_count[recv_proc]; k++ )
                {
                    gpu_sr[1] = j;
                    gpu_sr[3] = k;
                    printf("Test between %d, GPU %d and %d, GPU %d began\n", send_proc,
                    j, recv_proc, k);
                    MPI_Send( gpu_sr, 4, MPI_INT, send_proc, 1, MPI_COMM_WORLD );
                    MPI_Send( gpu_sr, 4, MPI_INT, recv_proc, 1, MPI_COMM_WORLD );

                    MPI_Recv( &conf, 1, MPI_INT, send_proc, 2, MPI_COMM_WORLD, &status );
                    MPI_Recv( &conf, 1, MPI_INT, recv_proc, 2, MPI_COMM_WORLD, &status );
            printf("Test between %d, GPU %d and %d, GPU %d finished", send_proc,
                j, recv_proc, k);
                }

            }
        }
        gpu_sr[0] = -1;
        for ( i = 1; i < comm_size; i++ )
            MPI_Send( gpu_sr, 4, MPI_INT, i, 1, MPI_COMM_WORLD );
    }
    else 
    {
        while( 1 )
        {
            MPI_Recv( gpu_sr, 4, MPI_INT, 0, 1, MPI_COMM_WORLD, &status );
            send_proc = gpu_sr[0];
            send_gpu = gpu_sr[1];
            recv_proc = gpu_sr[2];
            recv_gpu = gpu_sr[3];

            if ( send_proc == -1 )
                break;
            if ( send_proc == comm_rank )
                real_one_to_one_cuda( times, mes_length, num_repeats, send_proc, recv_proc,
                                      send_gpu, recv_gpu );
            if ( recv_proc == comm_rank )
                real_one_to_one_cuda( times, mes_length, num_repeats, send_proc, recv_proc,
                                      send_gpu, recv_gpu );
            
            MPI_Send( &conf, 1, MPI_INT, 0, 2, MPI_COMM_WORLD );
        }
    }
    return 0;
}

void real_one_to_one_cuda( Test_time_result_type *times, int mes_length, int num_repeats, int source_proc, int dest_proc,
                           int source_gpu, int dest_gpu )
{
    px_my_time_type time_beg,time_end;
    char *data = NULL;
    char *dataGPU = NULL;
    px_my_time_type *tmp_results=NULL;
    MPI_Status status;
    int i;
    int tmp;
    int stride = 0;

    for ( i = 0; i < source_proc; i++) 
        stride += gpu_count[i];

    if ( source_proc == dest_proc )
    {
        if ( source_gpu == dest_gpu )
        {
            times[stride + source_gpu].average = 0;
            times[stride + source_gpu].deviation = 0;
            times[stride + source_gpu].median = 0;
            return;
        }
        else
        {
            float timing;
            tmp_results = ( px_my_time_type* )malloc( num_repeats * sizeof( px_my_time_type ) );
            cudaEvent_t start, stop;
            cudaStream_t src_dst_stream;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaSetDevice( source_gpu );
            cudaMalloc( ( void** ) &data, mes_length );
            cudaDeviceEnablePeerAccess ( dest_gpu, 0 );
            cudaStreamCreate ( &src_dst_stream );
            cudaSetDevice( dest_gpu );
            cudaMalloc( ( void** ) &dataGPU, mes_length );
            cudaSetDevice( source_gpu );
            for ( i = 0; i < num_repeats; i++ )
            {
                cudaEventRecord ( start, 0 );
                cudaMemcpyPeerAsync( dataGPU, dest_gpu, data, source_gpu, mes_length, src_dst_stream );
                cudaEventRecord ( stop, 0 );
                cudaDeviceSynchronize();
                cudaEventElapsedTime ( &timing, start, stop );
                tmp_results[i] = (double)timing * 0.0001;
            }
            cudaFree ( ( void** ) &data);
            cudaDeviceReset();
            cudaSetDevice( dest_gpu );
            cudaFree ( ( void** ) &dataGPU);
            cudaDeviceReset();
            times[stride + source_gpu] = calc_stats( tmp_results, num_repeats );
            printf("Test between %d:%d and %d:%d finished with %lf med, %lf dev and %lf avg\n",
                    source_proc, source_gpu, dest_proc, dest_gpu, times[stride + source_gpu].median,
                    times[stride + source_gpu].deviation, times[stride + source_gpu].average);
            free ( tmp_results );
            return;
        }
    }

    tmp_results = ( px_my_time_type* )malloc( num_repeats * sizeof( px_my_time_type ) );

    if ( comm_rank == source_proc )
        cudaSetDevice( source_gpu );
    if ( comm_rank == dest_proc )
        cudaSetDevice( dest_gpu );

    cudaMalloc( ( void** ) &dataGPU, mes_length );
    data = ( char* )malloc( sizeof( char ) * mes_length );

    for ( i = 0; i < num_repeats; i++ )
    {
        if ( comm_rank == source_proc )
        {
            // cudaMemcpy to host
            time_beg = px_my_cpu_time();
            cudaMemcpy( dataGPU, data, mes_length, cudaMemcpyDeviceToHost );
            MPI_Send( data, mes_length, MPI_BYTE, dest_proc, 0, MPI_COMM_WORLD);
            time_end = px_my_cpu_time();
            tmp_results[i] = ( time_end - time_beg );
            MPI_Recv( &tmp, 1, MPI_INT, dest_proc, 100, MPI_COMM_WORLD, &status );
        }
        if ( comm_rank == dest_proc )
        {
            time_beg = px_my_cpu_time();
            MPI_Recv( data, mes_length, MPI_BYTE, source_proc, 0, MPI_COMM_WORLD, &status);
            cudaMemcpy( data, dataGPU, mes_length, cudaMemcpyHostToDevice );
            time_end = px_my_cpu_time();
            tmp_results[i] = ( time_end - time_beg );
            MPI_Send( &comm_rank, 1, MPI_INT, source_proc, 100, MPI_COMM_WORLD );
        }
    }

    cudaFree ( (void**) &dataGPU );
    cudaDeviceReset ();
    free( data );
 
    if ( source_proc == comm_rank ) 
    {
        free( tmp_results );
        return;
    }
    times[stride + source_gpu] = calc_stats( tmp_results, num_repeats );
    printf("Test between %d:%d and %d:%d finished with %lf med, %lf dev and %lf avg\n",
                    source_proc, source_gpu, dest_proc, dest_gpu, times[stride + source_gpu].median,
                    times[stride + source_gpu].deviation, times[stride + source_gpu].average);
}

Test_time_result_type calc_stats( px_my_time_type* all_times, int num_repeats )
{
    int i;
    Test_time_result_type  times;
    px_my_time_type sum = 0;
    for(i=0; i<num_repeats; i++)
    {
        sum+=all_times[i];
    }
    times.average=(sum/(double)num_repeats);

    px_my_time_type st_deviation = 0;
    for(i=0; i<num_repeats; i++)
    {
        st_deviation+=(all_times[i]-times.average)*(all_times[i]-times.average);
    }
    st_deviation/=(double)(num_repeats);
    times.deviation=sqrt(st_deviation);

    qsort(all_times, num_repeats, sizeof(px_my_time_type), my_time_cmp );
    times.median=all_times[num_repeats/2];

    times.min=all_times[0];

    return times;
}
