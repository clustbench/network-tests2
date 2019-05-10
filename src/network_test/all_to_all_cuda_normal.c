#include "my_time.h"
#include "my_malloc.h"
#include "tests_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <pthread.h>
#include <cuda_runtime.h>

extern int comm_rank;
extern int comm_size;
extern int* gpu_count;
extern int total_gpu;

int all_to_all_cuda( Test_time_result_type * times, int mes_length, int num_repeats );

extern Test_time_result_type calc_stats( px_my_time_type* all_times, int num_repeats );

int all_to_all_cuda( Test_time_result_type * times, int mes_length, int num_repeats )
{


    px_my_time_type **tmp_results=NULL;
    px_my_time_type time_beg,time_end;
    
    MPI_Status status;
    MPI_Request *send_request=NULL;
    MPI_Request *recv_request=NULL;
    int finished;
    px_my_time_type st_deviation;
    int i,j;
    int l_bound = 0, r_bound = 0;
    int flag=0;
    int tag = 0;
    double sum;

    int *gpu_mpi_rank;

    gpu_mpi_rank = ( int* )malloc( sizeof( int ) * total_gpu );
    int k = 0;
    for ( i = 0; i < comm_size; i++ ) 
    {
        while ( k < gpu_count[i] )
        {
            gpu_mpi_rank[k] = i;
            k++;
        }
    }


    int stride = 0;
    for ( i = 0 ; i < comm_rank; i++)
    {
        l_bound += gpu_count[i];
        r_bound += gpu_count[i];
    }
    r_bound += gpu_count[comm_rank] - 1;
    char **send_data=NULL;
    char **recv_data=NULL;

    char **send_data_host = NULL;
    char **recv_data_host = NULL;

    cudaStream_t *send_streams;
    cudaStream_t *recv_streams;

    cudaEvent_t *start_events;
    cudaEvent_t *stop_events;
    
    send_streams = ( cudaStream_t* )malloc( sizeof( cudaStream_t ) * total_gpu * gpu_count[comm_rank] );
    recv_streams = ( cudaStream_t* )malloc( sizeof( cudaStream_t ) * total_gpu * gpu_count[comm_rank] );
    
    start_events = ( cudaEvent_t* )malloc( sizeof( cudaEvent_t ) * total_gpu * gpu_count[comm_rank] );
    stop_events = ( cudaEvent_t* )malloc( sizeof( cudaEvent_t ) * total_gpu * gpu_count[comm_rank] );

    send_request = ( MPI_Request* )malloc( sizeof( MPI_Request ) *  total_gpu * gpu_count[comm_rank] );
    recv_request = ( MPI_Request* )malloc( sizeof( MPI_Request ) *  total_gpu * gpu_count[comm_rank] );
    /*cudaEvent_t start, stop;
    cudaStream_t src_dst_stream; //gpu_count[comm_rank] * gpu_count[comm_rank]

    cudaEventCreate( &start );
    cudaEventCreate( &stop ); */


    send_data = ( char** ) malloc ( sizeof( char* ) * total_gpu * gpu_count[comm_rank] );
    recv_data = ( char** ) malloc ( sizeof( char* ) * total_gpu * gpu_count[comm_rank] );

    send_data_host = ( char** ) malloc ( sizeof( char* ) * total_gpu * gpu_count[comm_rank] );
    recv_data_host = ( char** ) malloc ( sizeof( char* ) * total_gpu * gpu_count[comm_rank] );

    for ( i = 0; i < total_gpu * gpu_count[comm_rank]; i++ )
    {
        send_data_host[i] = ( char* )malloc( sizeof ( char ) * mes_length );
        recv_data_host[i] = ( char* )malloc( sizeof ( char ) * mes_length );
    }

    tmp_results = ( px_my_time_type** )malloc( sizeof( px_my_time_type* ) * total_gpu * gpu_count[comm_rank] );
    for ( i = 0; i < total_gpu * gpu_count[comm_rank]; i++ )
        tmp_results[i] = ( px_my_time_type* )malloc( sizeof( px_my_cpu_time ) * num_repeats );


    for ( i = 0; i < gpu_count[comm_rank]; i++ ) 
    {
        cudaSetDevice( i );
        for ( j = 0; j < total_gpu; j++ ) 
        {
            cudaMalloc( ( void** ) &send_data[i * total_gpu + j], mes_length );
            cudaMalloc( ( void** ) &recv_data[i * total_gpu + j], mes_length );

            cudaStreamCreate ( &send_streams[i * total_gpu + j] );
            cudaStreamCreate ( &recv_streams[i * total_gpu + j] );

            cudaEventCreate ( &start_events[i * total_gpu + j] );
            cudaEventCreate ( &stop_events[i * total_gpu + j] );
            //cudaMalloc( ( void** ) &data, mes_length );
        }

        for ( j = 0; j < gpu_count[comm_rank]; j++ )
        {
          //  if ( i == j )
          //     continue;
            cudaDeviceEnablePeerAccess ( j, 0 );
        }
    } //preparations


    //time_beg?
    for ( i = 0; i < num_repeats; i++ )
    {
        for ( j = 0; j < gpu_count[comm_rank]; j++ )
        {
            cudaSetDevice( j );
            for ( k = 0; k < total_gpu; k++ )
            {
                if ( comm_rank == gpu_mpi_rank[k] ) 
                {
                    cudaEventRecord( start_events[j * total_gpu + k], send_streams[j * total_gpu + k] );
                    cudaMemcpyPeerAsync( recv_data[k - r_bound], k - r_bound, send_data[j], j, mes_length, send_streams[j * total_gpu + k] );
                    cudaEventRecord( stop_events[j * total_gpu + k], send_streams[j * total_gpu + k]);
                    continue;
                   //part with memcpypeerasync on one board
                }
                else
                {
                
                   cudaMemcpyAsync( send_data_host[j * total_gpu + k], send_data[j * total_gpu + k], mes_length, cudaMemcpyDeviceToHost, send_streams[j * total_gpu + k] );
                   //cudaDeviceSynchronize();
                   //MPI_Isend(send_data[j], gpu_mpi_rank[j], mes_length, MPI_BYTE, );
                   //MPI_Irecv();

                }
            }

        }
        for ( j = 0; j < gpu_count[comm_rank]; j++ )
            for ( k = 0; k < total_gpu; k++ )
            {
                if ( comm_rank == gpu_mpi_rank[j] ) 
                    continue;
                cudaSetDevice( j );
                cudaStreamSynchronize( send_streams[j * total_gpu + k] );
                tag = (k << 24) | (k << 16);
                printf("%d %d %d\n", j, k, tag);
                MPI_Isend( send_data_host[j * total_gpu + k], mes_length, MPI_BYTE, gpu_mpi_rank[j], tag, MPI_COMM_WORLD, &send_request[j * total_gpu + k] );
                MPI_Isend( recv_data_host[j * total_gpu + k], mes_length, MPI_BYTE, gpu_mpi_rank[j], tag, MPI_COMM_WORLD, &recv_request[j * total_gpu + k] );
            }
        for ( j = 0; j < (total_gpu - gpu_count[comm_rank] ) * gpu_count[comm_rank]; j++ ) 
        {
                
                MPI_Waitany( total_gpu * gpu_count[comm_rank], recv_request, &finished, &status );
                int gpu_recv = ( status.MPI_TAG >> 24 );
                int gpu_send = ( ( status.MPI_TAG & 0xFF0000000 ) >> 16 );
                cudaSetDevice( gpu_recv );
                cudaMemcpyAsync( recv_data_host[gpu_recv * total_gpu + gpu_send], recv_data[gpu_recv], mes_length, cudaMemcpyHostToDevice, recv_streams[gpu_recv * total_gpu + gpu_send] );
                //cudaEventRecord();
                cudaStreamSynchronize( recv_streams[gpu_recv * total_gpu + gpu_send] );
                time_end = px_my_cpu_time();
               // tmp_results[finished][j] = time_end - time_beg;
        }
    
    }
    return 0;
}

void real_all_to_all_cuda ( Test_time_result_type * times, int mes_length, int num_repeats)
{
    return;
}

