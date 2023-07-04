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

#include "my_time.h"
#include "tests_common.h"
#include "benchmarks_common.h"

#define UNKNOWN_FLAG 3 /*use for every benchmark*/

//extern int comm_rank;
//extern int comm_size;

Test_time_result_type real_one_to_one(int mes_length,int num_repeats,int source_proc,int dest_proc,clustbench_time_t *real_times,int comm_size, int comm_rank);

static int random_option_1_id, random_option_2_id;

static int *random_option_1 = NULL;
static int *random_option_2 = NULL;

static int random_option_1_default = 0;
static int random_option_2_default = 3;

static int frequency = 1000;
static int window_amount = 100;
static int window_sum_length = 10;
static double k_min = 0.99;
static double k_avg = 0.99;
static double k_disp = 0.99;
static double k_med = 0.99;
double k_stats[4] = {k_min, k_avg, k_disp, k_med};


char *one_to_one_short_description = "short description";


int one_to_one(Test_time_result_type *times, clustbench_time_t *real_times, int mes_length,int num_repeats)
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
            send_proc=get_send_processor(i,comm_size);
            recv_proc=get_recv_processor(i,comm_size);

            pair[0]=send_proc;
            pair[1]=recv_proc;

            if(send_proc)
                MPI_Send(pair,2,MPI_INT,send_proc,1,MPI_COMM_WORLD);
            if(recv_proc)
                MPI_Send(pair,2,MPI_INT,recv_proc,1,MPI_COMM_WORLD);

            if(recv_proc==0)
            {
                times[send_proc]=real_one_to_one(mes_length,num_repeats,send_proc,recv_proc,real_times,comm_size,comm_rank);
            }
            if(send_proc==0)
            {
                real_one_to_one(mes_length,num_repeats,send_proc,recv_proc,real_times,comm_size,comm_rank);
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
                real_one_to_one(mes_length,num_repeats,send_proc,recv_proc,real_times,comm_size,comm_rank);
            if(recv_proc==comm_rank)
                times[send_proc]=real_one_to_one(mes_length,num_repeats,send_proc,recv_proc,real_times,comm_size,comm_rank);

            confirmation_flag=1;
            MPI_Send(&confirmation_flag,1,MPI_INT,0,1,MPI_COMM_WORLD);
        }
    } /* end else comm_rank==0 */

    return 0;
} /* end one_to_one */


Test_time_result_type real_one_to_one(int mes_length,int num_repeats,int source_proc,int dest_proc,clustbench_time_t *real_times,int comm_size, int comm_rank)
{
    px_my_time_type time_beg,time_end;
    char *data=NULL;
    px_my_time_type *tmp_results=NULL;
    MPI_Status status;
    int i;
    px_my_time_type sum;
    Test_time_result_type  times;

    px_my_time_type st_deviation;
    int tmp;

    if(source_proc==dest_proc)
    {
        times.average=0;
        times.deviation=0;
        times.median=0;
        return times;
    }

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


    int checker = num_repeats/frequency;

    for(i=0; i<num_repeats; i++)
    {
        if ((i != 0) && (i % checker == 0))
        {
            window_length = window_sum_length*cur_iter/window_amount;
            int max_stats[4] = {0, 0 ,0 ,0};
            int min_stats[4] = {0, 0, 0, 0};

            for (int win = 0; win < window_amount; win++)
            {
                cur_window = (px_my_time_type*)malloc(window_length*sizeof(px_my_time_type));
                srand(win);

                // {min, avg, disp, med}
                int win_stats[4] = {0, 0, 0, 0};

                for (int ind = 0; ind < window_length; ind++)
                {
                    int cur_num = tmp_results[rand()%i];
                    if ((win_stats[0] == 0) || (cur_num < win_stats[0]))
                    {
                        win_stats[0] = cur_num;
                    }
                    win_stats[1] = win_stats[1]+cur_num;
                    win_stats[2] = win_stats[2]*cur_num*cur_num;
                    cur_window[ind] = cur_num;
                }

                win_stats[1] = win_stats[1]/window_length;
                win_stats[2] = win_stats[2]/window_length - win_stats[1]*win_stats[1];

                qsort(cur_window, window_length, sizeof(px_my_time_type), clustbench_time_cmp);
                win_stats[3] = cur_window[window_length/2];

                for (int stat_num = 0; stat_num < 4; stat_num++)
                {
                    if (win_stats[stat_num] > max_stats[stat_num])
                    {
                        max_stats[stat_num] = win_stats[stat_num];
                    }
                    if (win_stats[stat_num] < min_stats[stat_num])
                    {
                        min_stats[stat_num] = win_stats[stat_num];
                    }
                }
            }

            int flag_br = 1;
            for (int stat_num = 0; stat_num < 4; stat_num++)
            {
                if ((max_stats[stat_num]-min_stats[stat_num])/max_stats[stat_num] < k_stats[stat_num])
                {
                    flag_br = 0;
                    break;
                }
            }
            
            if (flag_br == 1)
            {
                break;
            }

        }
        if (i % frequency
        if(comm_rank==source_proc)
        {
            time_beg=px_my_cpu_time();

            MPI_Send(	data,
                        mes_length,
                        MPI_BYTE,
                        dest_proc,
                        0,
                        MPI_COMM_WORLD
                    );

            time_end=px_my_cpu_time();

            MPI_Recv(&tmp,1,MPI_INT,dest_proc,100,MPI_COMM_WORLD,&status);

            tmp_results[i]=(time_end-time_beg);
            /*
             printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
             comm_rank,comm_size,mes_length,i,tmp_results[i]);
            */
        }
        if(comm_rank==dest_proc)
        {

            time_beg=px_my_cpu_time();

            MPI_Recv(	data,
                        mes_length,
                        MPI_BYTE,
                        source_proc,
                        0,
                        MPI_COMM_WORLD,
                        &status
                    );


            time_end=px_my_cpu_time();
            tmp_results[i]=(time_end-time_beg);

            MPI_Send(&comm_rank,1,MPI_INT,source_proc,100,MPI_COMM_WORLD);
            /*
             printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
             comm_rank,comm_size,mes_length,finished,times[finished]);
            */
        }
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

    qsort(tmp_results, num_repeats, sizeof(px_my_time_type), clustbench_time_cmp );
    times.median=tmp_results[num_repeats/2];

    times.min=tmp_results[0];

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
