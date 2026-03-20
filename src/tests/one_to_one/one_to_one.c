/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Alexey N. Salnikov (salnikov@cmc.msu.ru)
 *
 */

#ifdef _GNU_SOURCE
#include <getopt.h>
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <math.h>
#include <mpi.h>
#include <errno.h>
#include <string.h>

#include "delay_measurements_amount_auto.h"
#include "my_time.h"
#include "tests_common.h"
#include "benchmarks_common.h"

#define UNKNOWN_FLAG 3 /*use for every benchmark*/
#define SRC_DEST_ORDER_FLAG 0 /*if equals 1, then src - external loop, if equals 0, then dest - external loop*/

//extern int comm_rank;
//extern int comm_size;

Test_time_result_type real_one_to_one(int mes_length,int num_repeats,int source_proc,int dest_proc,clustbench_time_t *real_times,int comm_size, int comm_rank, int timer_type, struct AlgorithmMainInfo* algorithm_main_info);
void real_one_to_one_mash(int amount_of_lengths,
                                            int num_repeats,
                                            int beg_length,
                                            int step_length,
                                            int end_length,
                                            int *permutation,
                                            int permutation_length,
                                            int source_proc,
                                            int dest_proc,
                                            clustbench_time_result_t **times,
                                            clustbench_time_t **real_times,
                                            int comm_size,
                                            int comm_rank,
                                            int timer_type);

static int random_option_1_id, random_option_2_id;

static int *random_option_1 = NULL;
static int *random_option_2 = NULL;

static int random_option_1_default = 0;
static int random_option_2_default = 3;

/*
static int frequency = 1000;
static int window_amount = 100;
static int window_sum_length = 10;
#define k_min 0.99
#define k_avg  0.99
#define k_disp  0.99
#define k_med  0.99
double k_stats[4] = {k_min, k_avg, k_disp, k_med};
*/


char *one_to_one_short_description = "short description";


int one_to_one(Test_time_result_type *times, clustbench_time_t *real_times, int mes_length,int num_repeats,int timer_type,struct AlgorithmMainInfo* algorithm_main_info)
{
    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
    
    int i;
    int pair[2];

    int confirmation_flag;

    int send_proc,recv_proc;

    MPI_Status status;

    if(comm_rank==0)
    {
        for(i=0; i<comm_size*comm_size; i++)
        {
            if (SRC_DEST_ORDER_FLAG == 1)
            {
                send_proc=get_send_processor(i,comm_size);
                recv_proc=get_recv_processor(i,comm_size);
            }
            else {
                send_proc=reverse_get_send_processor(i,comm_size);
                recv_proc=reverse_get_recv_processor(i,comm_size);
            }
            printf("Sending from %d to %d\n", send_proc, recv_proc);

            pair[0]=send_proc;
            pair[1]=recv_proc;

            if(send_proc)
                MPI_Send(pair,2,MPI_INT,send_proc,1,MPI_COMM_WORLD);
            if(recv_proc)
                MPI_Send(pair,2,MPI_INT,recv_proc,1,MPI_COMM_WORLD);

            if(recv_proc==0)
            {
                times[send_proc]=real_one_to_one(mes_length,num_repeats,send_proc,recv_proc,real_times,comm_size,comm_rank,timer_type,algorithm_main_info);
            }
            if(send_proc==0)
            {
                real_one_to_one(mes_length,num_repeats,send_proc,recv_proc,real_times,comm_size,comm_rank,timer_type,algorithm_main_info);
            }
            if(send_proc)
            {
                MPI_Recv(&confirmation_flag,1,MPI_INT,send_proc,1,MPI_COMM_WORLD,&status);
            }

            if(recv_proc)
            {
                MPI_Recv(&confirmation_flag,1,MPI_INT,recv_proc,1,MPI_COMM_WORLD,&status);
            }

        } /* End for */
        pair[0]=-1;
        for(i=1; i<comm_size; i++)
            MPI_Send(pair,2,MPI_INT,i,1,MPI_COMM_WORLD);
    } /* end if comm_rank==0 */
    else
    {
        for( ; ; )
        {
            MPI_Recv(pair,2,MPI_INT,0,1,MPI_COMM_WORLD,&status);
            send_proc=pair[0];
            recv_proc=pair[1];

	          if(send_proc==-1)
                break;
            if(send_proc==comm_rank)
                real_one_to_one(mes_length,num_repeats,send_proc,recv_proc,real_times,comm_size,comm_rank,timer_type,algorithm_main_info);
            if(recv_proc==comm_rank) {
                times[send_proc]=real_one_to_one(mes_length,num_repeats,send_proc,recv_proc,real_times,comm_size,comm_rank,timer_type,algorithm_main_info);
                if ((recv_proc == send_proc) && (recv_proc == 4)) {
                  printf("CHECK W %d\n", times[send_proc].measurements_amount);
                }
            }
            confirmation_flag=1;
            MPI_Send(&confirmation_flag,1,MPI_INT,0,1,MPI_COMM_WORLD);
        }
    } /* end else comm_rank==0 */
} /* end one_to_one */


Test_time_result_type real_one_to_one(int mes_length,int num_repeats,int source_proc,int dest_proc,clustbench_time_t *real_times,int comm_size, int comm_rank, int timer_type, struct AlgorithmMainInfo* algorithm_main_info)
{
    px_my_time_type time_beg,time_end;
    struct timespec time_beg_realtime, time_end_realtime;
    char *data=NULL;
    px_my_time_type *tmp_results=NULL;
    MPI_Status status;
    int i;
    px_my_time_type sum;
    Test_time_result_type times;

    px_my_time_type st_deviation;
    int tmp;

    if(source_proc==dest_proc)
    {
        times.average=0;
        times.deviation=0;
        times.median=0;
        times.min=0;
        times.measurements_amount=0;
        printf("DONE NULLS %d %d %d %d\n", comm_rank, source_proc, dest_proc, mes_length);
        return times;
    }
    printf("NOT NULLS %d %d %d %d\n", comm_rank, source_proc, dest_proc, mes_length);

    tmp_results=(px_my_time_type *)malloc(num_repeats*sizeof(px_my_time_type));
    if(tmp_results==NULL)
    {
        printf("proc %d from %d: Can not allocate memory\n",comm_rank,comm_size);
        times.average=-1;
        return times;
    }

    data=(char *)malloc(mes_length*sizeof(char));
    if(data==NULL)
    {
        free(tmp_results);
        printf("proc %d from %d: Can not allocate memory\n",comm_rank,comm_size);
        times.average=-1;
        return times;
    }

    if (algorithm_main_info->algorithm_general_type == kNoAlgo) {
        for(i=0; i<num_repeats; i++)
        {
            if(comm_rank==source_proc)
            {
              if (timer_type == 0) {
                time_beg=px_my_cpu_time();
              }
              else {
                clock_gettime(CLOCK_REALTIME, &time_beg_realtime);
              }

              MPI_Send(	data,
                        mes_length,
                        MPI_BYTE,
                        dest_proc,
                        0,
                        MPI_COMM_WORLD
                      );

              if (timer_type == 0) {
                time_end=px_my_cpu_time();
              }
              else {
                clock_gettime(CLOCK_REALTIME, &time_end_realtime);
              }

              MPI_Recv(&tmp,1,MPI_INT,dest_proc,100,MPI_COMM_WORLD,&status);

              if (timer_type == 0) {
                tmp_results[i]=(time_end-time_beg);
              }
              else {
                // 999999999 наносекунд в секунде.
                tmp_results[i] = (time_end_realtime.tv_sec-time_beg_realtime.tv_sec) + (double)(time_end_realtime.tv_nsec-time_beg_realtime.tv_nsec)/999999999;
              }
              /*
              printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
              comm_rank,comm_size,mes_length,i,tmp_results[i]);
              */
            }
            if(comm_rank==dest_proc)
            {

                if (timer_type == 0) {
                  time_beg=px_my_cpu_time();
                }
                else {
                  clock_gettime(CLOCK_REALTIME, &time_beg_realtime);
                }

                MPI_Recv(	data,
                        mes_length,
                        MPI_BYTE,
                        source_proc,
                        0,
                        MPI_COMM_WORLD,
                        &status
                    );


                if (timer_type == 0) {
                  time_end=px_my_cpu_time();
                }
                else {
                  clock_gettime(CLOCK_REALTIME, &time_end_realtime);
                }

                if (timer_type == 0) {
                  tmp_results[i]=(time_end-time_beg);
                }
                else {
                  // 999999999 наносекунд в секунде.
                  tmp_results[i] = (time_end_realtime.tv_sec-time_beg_realtime.tv_sec) + (double)(time_end_realtime.tv_nsec-time_beg_realtime.tv_nsec)/999999999;
                }

                MPI_Send(&comm_rank,1,MPI_INT,source_proc,100,MPI_COMM_WORLD);
                /*
                 printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
                 comm_rank,comm_size,mes_length,finished,times[finished]);
                */
            }
        }
        if (comm_rank == source_proc) {
          free(data);
          free(tmp_results);
          times.average = 0;
          times.deviation = 0;
          times.median = 0;
          times.measurements_amount = 0;
          return times;
        }

        sum=0;
        for(i=0; i<num_repeats; i++)
        {
          sum+=tmp_results[i];
          real_times[source_proc*num_repeats + i] = tmp_results[i];
        }
        times.average=(sum/(double)num_repeats);

        st_deviation=0;
        for(i=0; i<num_repeats; i++)
        {
          st_deviation+=(tmp_results[i]-times.average)*(tmp_results[i]-times.average);
        }
        st_deviation/=(double)(num_repeats);
        times.deviation=sqrt(st_deviation);

        qsort(tmp_results, num_repeats, sizeof(px_my_time_type), clustbench_time_cmp);
        times.median=tmp_results[num_repeats/2];

        times.min=tmp_results[0];
    } // END if (algorithm_main_info->algorithm_general_type == kNoAlgo)
    else {
        int i = 0;
        while (i < num_repeats) {
            for (int j = 0; j < algorithm_main_info->frequency; j++) {
              if(comm_rank==source_proc)
              {
                if (timer_type == 0) {
                  time_beg=px_my_cpu_time();
                }
                else {
                  clock_gettime(CLOCK_REALTIME, &time_beg_realtime);
                }

                MPI_Send(	data,
                        mes_length,
                        MPI_BYTE,
                        dest_proc,
                        0,
                        MPI_COMM_WORLD
                      );

                if (timer_type == 0) {
                  time_end=px_my_cpu_time();
                }
                else {
                  clock_gettime(CLOCK_REALTIME, &time_end_realtime);
                }

                MPI_Recv(&tmp,1,MPI_INT,dest_proc,100,MPI_COMM_WORLD,&status);

                if (timer_type == 0) {
                  tmp_results[i]=(time_end-time_beg);
                }
                else {
                  // 999999999 наносекунд в секунде.
                  tmp_results[i] = (time_end_realtime.tv_sec-time_beg_realtime.tv_sec) + (double)(time_end_realtime.tv_nsec-time_beg_realtime.tv_nsec)/999999999;
                }
                /*
                printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
                comm_rank,comm_size,mes_length,i,tmp_results[i]);
                */
              }
              if(comm_rank==dest_proc)
              {

                if (timer_type == 0) {
                  time_beg=px_my_cpu_time();
                }
                else {
                  clock_gettime(CLOCK_REALTIME, &time_beg_realtime);
                }

                MPI_Recv(	data,
                        mes_length,
                        MPI_BYTE,
                        source_proc,
                        0,
                        MPI_COMM_WORLD,
                        &status
                    );


                if (timer_type == 0) {
                  time_end=px_my_cpu_time();
                }
                else {
                  clock_gettime(CLOCK_REALTIME, &time_end_realtime);
                }

                if (timer_type == 0) {
                  tmp_results[i]=(time_end-time_beg);
                }
                else {
                  // 999999999 наносекунд в секунде.
                  tmp_results[i] = (time_end_realtime.tv_sec-time_beg_realtime.tv_sec) + (double)(time_end_realtime.tv_nsec-time_beg_realtime.tv_nsec)/999999999;
                }

                MPI_Send(&comm_rank,1,MPI_INT,source_proc,100,MPI_COMM_WORLD);
                /*
                 printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
                 comm_rank,comm_size,mes_length,finished,times[finished]);
                */
              }
              i++;
            }

            int continue_flag = 1;
            if (comm_rank == source_proc)
            {
              MPI_Recv(&continue_flag, 1, MPI_INT, dest_proc, 111, MPI_COMM_WORLD, &status);
            }
            if (comm_rank == dest_proc)
            {
              double cur_criterion_value;
              if (algorithm_main_info->algorithm_general_type == kScalarAlgo) {
                  cur_criterion_value = scalar_criterion(i, algorithm_main_info->window_selection_parameters, algorithm_main_info->scalar_algorithm_counter, tmp_results);
              }
              else if (algorithm_main_info->algorithm_general_type == kSpectrumAlgo) {
                  cur_criterion_value = spectrum_criterion(i, algorithm_main_info->window_selection_parameters, algorithm_main_info->spectrum_algorithm_counters, tmp_results);
              }
              if (cur_criterion_value > algorithm_main_info->target_criterion_value) {
                  continue_flag = 0;
              }
              MPI_Send(&continue_flag, 1, MPI_INT, source_proc, 111, MPI_COMM_WORLD);
            }

            if (continue_flag == 0) {
              break;
            }
        }
        int amount_of_iterations = i;

        if (comm_rank == source_proc) {
          free(data);
          free(tmp_results);
          times.average = -2;
          times.deviation = -2;
          times.median = -2;
          return times;
        }

        sum=0;
        for(i=0; i<amount_of_iterations; i++)
        {
          sum+=tmp_results[i];
          real_times[source_proc*num_repeats + i] = tmp_results[i];
        }
        for(i=amount_of_iterations; i<num_repeats; i++) {
          real_times[source_proc*num_repeats + i] = -1;
        }
        times.average=(sum/(double)amount_of_iterations);

        st_deviation=0;
        for(i=0; i<amount_of_iterations; i++)
        {
          st_deviation+=(tmp_results[i]-times.average)*(tmp_results[i]-times.average);
        }
        st_deviation/=(double)(amount_of_iterations);
        times.deviation=sqrt(st_deviation);

        qsort(tmp_results, amount_of_iterations, sizeof(px_my_time_type), clustbench_time_cmp);
        times.median=tmp_results[amount_of_iterations/2];

        times.min=tmp_results[0];
        times.measurements_amount = amount_of_iterations;

    }

    free(data);
    free(tmp_results);

    if((comm_rank==source_proc)||(comm_rank==dest_proc)) return times;
    else
    {
        times.average=-1;
        times.min=0;
        return times;
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

int one_to_one_mash(clustbench_time_result_t **times, clustbench_time_t **real_times,
                                int num_repeats,
                                int beg_length,
                                int step_length,
                                int end_length,
                                int permutation_length,
                                int amount_of_lengths,
                                int *permutation,
                                int timer_type)
{
    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
    
    int i;
    int pair[2];

    int confirmation_flag;

    int send_proc,recv_proc;

    MPI_Status status;
    if (comm_rank == 0)
    {
        for(i=0; i<comm_size*comm_size; i++)
        {
            if (SRC_DEST_ORDER_FLAG == 1) {
                send_proc=get_send_processor(i,comm_size);
                recv_proc=get_recv_processor(i,comm_size);
            }
            else {
                send_proc=reverse_get_send_processor(i,comm_size);
                recv_proc=reverse_get_recv_processor(i,comm_size);
            }
            printf("Sending from %d to %d\n", send_proc, recv_proc);

            pair[0]=send_proc;
            pair[1]=recv_proc;

            if(send_proc)
                MPI_Send(pair,2,MPI_INT,send_proc,1,MPI_COMM_WORLD);
            if(recv_proc)
                MPI_Send(pair,2,MPI_INT,recv_proc,1,MPI_COMM_WORLD);
            if(recv_proc==0)
            {
                real_one_to_one_mash(amount_of_lengths, num_repeats, beg_length, step_length, end_length, permutation, permutation_length, send_proc, recv_proc, times, real_times, comm_size, comm_rank, timer_type);
            }
            if(send_proc==0)
            {
                real_one_to_one_mash(amount_of_lengths, num_repeats, beg_length, step_length, end_length, permutation, permutation_length, send_proc, recv_proc, times, real_times, comm_size, comm_rank, timer_type);
            }
            if(send_proc)
            {
                MPI_Recv(&confirmation_flag,1,MPI_INT,send_proc,1,MPI_COMM_WORLD,&status);
            }

            if(recv_proc)
            {
                MPI_Recv(&confirmation_flag,1,MPI_INT,recv_proc,1,MPI_COMM_WORLD,&status);
            }

        } /* End for */
        pair[0]=-1;
        for(i=1; i<comm_size; i++)
            MPI_Send(pair,2,MPI_INT,i,1,MPI_COMM_WORLD);
    } /* end if comm_rank==0 */
    else {
        for( ; ; )
        {
            MPI_Recv(pair,2,MPI_INT,0,1,MPI_COMM_WORLD,&status);
            send_proc=pair[0];
            recv_proc=pair[1];

	        if(send_proc==-1)
                break;
            if(send_proc==comm_rank)
                real_one_to_one_mash(amount_of_lengths, num_repeats, beg_length, step_length, end_length, permutation, permutation_length, send_proc, recv_proc, times, real_times, comm_size, comm_rank, timer_type);
            if(recv_proc==comm_rank)
                real_one_to_one_mash(amount_of_lengths, num_repeats, beg_length, step_length, end_length, permutation, permutation_length, send_proc, recv_proc, times, real_times, comm_size, comm_rank, timer_type);

            confirmation_flag=1;
            MPI_Send(&confirmation_flag,1,MPI_INT,0,1,MPI_COMM_WORLD);
        }
    }
    return 0;
} /* end one_to_one_mash */

void real_one_to_one_mash(int amount_of_lengths,
                                            int num_repeats,
                                            int beg_length,
                                            int step_length,
                                            int end_length,
                                            int *permutation,
                                            int permutation_length,
                                            int source_proc,
                                            int dest_proc,
                                            clustbench_time_result_t **times,
                                            clustbench_time_t **real_times,
                                            int comm_size,
                                            int comm_rank,
                                            int timer_type)
{
    px_my_time_type time_beg,time_end;
    struct timespec time_beg_realtime, time_end_realtime;
    char *data=NULL;
    px_my_time_type *tmp_results=NULL;
    MPI_Status status;
    int i;
    px_my_time_type sum;
    
    px_my_time_type st_deviation;
    int tmp;

    printf("CKECING MEMORY IN REAL_ONE_TO_OEN_MASH %d\n", comm_rank);
    times[0][source_proc].median=0;

    if(source_proc==dest_proc)
    {
        printf("Starting to fill vector for equal processes: %d\n", source_proc);
        for (int i = 0; i < amount_of_lengths; i++) {
            times[i][source_proc].average=0;
            times[i][source_proc].deviation=0;
            times[i][source_proc].median=0;
            times[i][source_proc].min=0;
            times[i][source_proc].amount_of_measurements=0;
            printf("FILLED FOR %d MESSAGE LENGTH", i);
        }
        return;
    }

    tmp_results=(px_my_time_type *)malloc(permutation_length*sizeof(px_my_time_type));
    if(tmp_results==NULL)
    {
        printf("proc %d from %d: Can not allocate memory\n",comm_rank,comm_size);
        for (int i = 0; i < amount_of_lengths; i++) {
          times[i][source_proc].average=-1;
        }
        return;
    }

    data=(char *)malloc(end_length*sizeof(char));
    if(data==NULL)
    {
        free(tmp_results);
        printf("proc %d from %d: Can not allocate memory\n",comm_rank,comm_size);
        for(int i = 0; i < amount_of_lengths; i++) {
          times[i][source_proc].average=-1;
        }
        return;
    }

    if(comm_rank==source_proc)
    { 
        for (i = 0; i < permutation_length; i++) {
            if (timer_type == 0) {
              time_beg=px_my_cpu_time();
            }
            else {
              clock_gettime(CLOCK_REALTIME, &time_beg_realtime);
            }

            MPI_Send(	data,
                        permutation[i],
                        MPI_BYTE,
                        dest_proc,
                        0,
                        MPI_COMM_WORLD
            );

            if (timer_type == 0) {
              time_end=px_my_cpu_time();
            }
            else {
              clock_gettime(CLOCK_REALTIME, &time_end_realtime);
            }

            MPI_Recv(&tmp,1,MPI_INT,dest_proc,100,MPI_COMM_WORLD,&status);

            if (timer_type == 0) {
              tmp_results[i]=(time_end-time_beg);
            }
            else {
              // 999999999 наносекунд в секунде.
              tmp_results[i] = (time_end_realtime.tv_sec-time_beg_realtime.tv_sec) + (double)(time_end_realtime.tv_nsec-time_beg_realtime.tv_nsec)/999999999;
            }
        }
    }
    if(comm_rank==dest_proc) {
        for (i = 0; i < permutation_length; i++) {
            if (timer_type == 0) {
              time_beg=px_my_cpu_time();
            }
            else {
              clock_gettime(CLOCK_REALTIME, &time_beg_realtime);
            }

            MPI_Recv(	data,
                        permutation[i],
                        MPI_BYTE,
                        source_proc,
                        0,
                        MPI_COMM_WORLD,
                        &status
                    );


            if (timer_type == 0) {
              time_end=px_my_cpu_time();
            }
            else {
              clock_gettime(CLOCK_REALTIME, &time_end_realtime);
            }

            if (timer_type == 0) {
              tmp_results[i]=(time_end-time_beg);
            }
            else {
              // 999999999 наносекунд в секунде.
              tmp_results[i] = (time_end_realtime.tv_sec-time_beg_realtime.tv_sec) + (double)(time_end_realtime.tv_nsec-time_beg_realtime.tv_nsec)/999999999;
            }

            MPI_Send(&comm_rank,1,MPI_INT,source_proc,100,MPI_COMM_WORLD);
        }
    }
    free(data);

    if (comm_rank == source_proc) {
        free(tmp_results);
        return;
    }

    printf("Starting to fill vectors %d\n", comm_rank);

    int *filler_vector = (int*)malloc(amount_of_lengths*sizeof(int));
    for (int i = 0; i < amount_of_lengths; i++) {
        filler_vector[i] = 0;
    }
    for (int i = 0; i < permutation_length; i++) {
        int cur_length_ind = (permutation[i]-beg_length)/step_length;
        real_times[cur_length_ind][source_proc*num_repeats+filler_vector[cur_length_ind]] = tmp_results[i]; 
        filler_vector[cur_length_ind]++;
    }
    printf("FILLED REAL TIMES, SUCCESS %d\n", comm_rank);
    free(filler_vector);
    free(tmp_results);
    
    tmp_results = (clustbench_time_t *)malloc(num_repeats*sizeof(clustbench_time_t));
    for (int cur_length_ind = 0; cur_length_ind < amount_of_lengths; cur_length_ind++)
    {
        sum=0;
        for(i=0; i<num_repeats; i++)
        {
            tmp_results[i] = real_times[cur_length_ind][source_proc*num_repeats + i];
            sum += real_times[cur_length_ind][source_proc*num_repeats + i];
        }
        times[cur_length_ind][source_proc].average=(sum/(double)num_repeats);

        st_deviation=0;
        for(i=0; i<num_repeats; i++)
        {
            st_deviation+=(tmp_results[i]-times[cur_length_ind][source_proc].average)*
                          (tmp_results[i]-times[cur_length_ind][source_proc].average);
        }
        st_deviation/=(double)(num_repeats);
        times[cur_length_ind][source_proc].deviation=sqrt(st_deviation);

        qsort(tmp_results, num_repeats, sizeof(clustbench_time_t), clustbench_time_cmp);
        times[cur_length_ind][source_proc].median=tmp_results[num_repeats/2];

        times[cur_length_ind][source_proc].min=tmp_results[0];
    }

    free(tmp_results);

    if((comm_rank==source_proc)||(comm_rank==dest_proc)) return;
    else
    {
        for (int i = 0; i < amount_of_lengths; i++) {
            times[i][source_proc].average=-1;
            times[i][source_proc].min=0;
        }
        return;
    }
}


int one_to_one_print_help (clustbench_benchmark_parameters_t* parameters) 
{
    printf("This is help message of one_to_one test\n");
    printf("'f' for first arg, 's' for second\n\n");
    return 0;
}

int one_to_one_define_netcdf_vars (int file_id, clustbench_benchmark_parameters_t* params) 
{
    if(nc_def_var(file_id,"random_option_1",NC_INT,0,0,&random_option_1_id)!=NC_NOERR)
    {
            return 1;
    }
    if(nc_def_var(file_id,"random_option_2",NC_INT,0,0,&random_option_2_id)!=NC_NOERR)
    {
            return 1;
    }
    return 0;
}

int one_to_one_put_netcdf_vars (int file_id, clustbench_benchmark_parameters_t* parameters) 
{
    random_option_1 = (int *)parameters->benchmark_parameters;
    random_option_2 = (int *)(parameters->benchmark_parameters) + 1;
    
    if(nc_put_var_int(file_id,random_option_1_id,random_option_1)!=NC_NOERR)
	{
		return 1;
	}
    if(nc_put_var_int(file_id,random_option_2_id,random_option_2)!=NC_NOERR)
	{
		return 1;
	}
    return 0;
}

int one_to_one_print_parameters (clustbench_benchmark_parameters_t* parameters) 
{
    random_option_1 = (int *)parameters->benchmark_parameters;
    random_option_2 = (int *)(parameters->benchmark_parameters) + 1;
    printf("\trandom_option_1 = %d\n", *random_option_1);
    printf("\trandom_option_2 = %d\n", *random_option_2);
    return 0;
}

int one_to_one_parse_parameters (clustbench_benchmark_parameters_t* parameters,int argc,char **argv,int mpi_rank) 
{
    parameters->benchmark_parameters = (void *) malloc(sizeof(int) * 2);
    if (parameters->benchmark_parameters == NULL)
    {
        printf("Can`t allocate memory for one_to_one test individual parameters\n");
        return 2;
    }
    
    random_option_1 = (int *)parameters->benchmark_parameters;
    random_option_2 = (int *)(parameters->benchmark_parameters) + 1;
    
    *random_option_1 = random_option_1_default;
    *random_option_2 = random_option_2_default;
    
    #ifdef _GNU_SOURCE
        struct option options[]=
        {
            {"first",required_argument,NULL,'a'},
            {"second",required_argument,NULL,'c'},
            {0,0,0,0}
        };
    #endif
    
    int arg_val;
    
    for ( ; ; )
    {
        #ifdef _GNU_SOURCE
                arg_val = getopt_long(argc,argv,"a:c:",options,NULL);
        #else
        
                arg_val = getopt(argc,argv,"a:c:");
        #endif
        
        if(arg_val == -1)
        break;

        switch(arg_val)
        {
            char *tmp_str;
        case 'a':
            *random_option_1 = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str != '\0')
            {
                if(!mpi_rank)
                {
                    fprintf(stderr,"one_to_one_parse_parameters: Parse parameter with name 'first' failed: '%s'",
                        strerror(errno));
                }
                return 1;
            }
            break;
        case 'c':
            *random_option_2 = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str != '\0')
            {
                if(!mpi_rank)
                {
                    fprintf(stderr,"one_to_one_parse_parameters: Parse parameter with name 'second' failed: '%s'",
                        strerror(errno));
                }
                return 1;
            }
            break;
        case '?':
            if(!mpi_rank)
            {
                one_to_one_print_help(NULL);
            }
            return UNKNOWN_FLAG; /* flag of the unknown option */
            break;
        }
    }
    return 0;
}

int one_to_one_free_parameters (clustbench_benchmark_parameters_t* parameters)
{
    free(parameters->benchmark_parameters);
    return 0;
}
