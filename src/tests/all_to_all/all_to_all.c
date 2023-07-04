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

#include "my_time.h"
#include "benchmarks_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <math.h>
#include <mpi.h>
#include <errno.h>
#include <string.h>

#define UNKNOWN_FLAG 3 /*use for every benchmark*/

/*extern int comm_rank;
extern int comm_size;*/

static int random_option_1_id, random_option_2_id;

static int *random_option_1 = NULL;
static int *random_option_2 = NULL;

static int random_option_1_default = 0;
static int random_option_2_default = 3;

char *all_to_all_short_description = "short description";

int all_to_all(clustbench_time_result_t *times,  clustbench_time_t *real_times, int mes_length,int num_repeats, void *additional)
{
    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
    
    px_my_time_type **tmp_results=NULL;
    px_my_time_type time_beg,time_end;
    char **send_data=NULL;
    char **recv_data=NULL;
    MPI_Status status;
    MPI_Request *send_request=NULL;
    MPI_Request *recv_request=NULL;
    int finished;
    px_my_time_type st_deviation;
    int i,j;
    int flag=0;
    double sum;


    tmp_results=(px_my_time_type**)malloc(comm_size*sizeof(px_my_time_type*));
    if(tmp_results==NULL)
    {
        free(times);
        free(real_times);
        return -1;
    }

    send_request=(MPI_Request *)malloc(comm_size*sizeof(MPI_Request));
    if(send_request == NULL)
    {
        free(times);
        free(real_times);
        free(tmp_results);
        return -1;
    }

    recv_request=(MPI_Request *)malloc(comm_size*sizeof(MPI_Request));
    if(recv_request == NULL)
    {
        free(times);
        free(real_times);
        free(tmp_results);
        free(send_request);
        return -1;
    }
    send_data=(char **)malloc(sizeof(char *)*comm_size);
    if(send_data == NULL)
    {
        free(times);
        free(tmp_results);
        free(send_request);
        free(recv_request);
        return -1;
    }
    recv_data=(char **)malloc(sizeof(char *)*comm_size);
    if(recv_data == NULL)
    {
        free(times);
        free(real_times);
        free(tmp_results);
        free(send_request);
        free(recv_request);
        free(send_data);
        return -1;
    }


    for(i=0; i<comm_size; i++)
    {
        send_data[i]=NULL;
        recv_data[i]=NULL;
        tmp_results[i]=NULL;

        tmp_results[i]=(px_my_time_type *)malloc(num_repeats*sizeof(px_my_time_type));
        if(tmp_results[i]==NULL)
        {
            flag=1;
        }

        send_data[i]=(char *)malloc(mes_length*sizeof(char));
        if(send_data[i]==NULL)
        {
            flag=1;
        }

        recv_data[i]=(char *)malloc(mes_length*sizeof(char));
        if(recv_data[i] == NULL)
        {
            flag=1;
        }
    }

    if(flag == 1)
    {
        free(times);
        free(real_times);
        free(send_request);
        free(recv_request);
        for(i=0; i<comm_size; i++)
        {
            if(send_data[i]!=NULL)   free(send_data[i]);
            if(recv_data[i]!=NULL)   free(recv_data[i]);
            if(tmp_results[i]!=NULL) free(tmp_results[i]);
        }
        free(send_data);
        free(recv_data);
        free(tmp_results);
        return -1;
    }

    for(i=0; i<num_repeats; i++)
    {

        time_beg=px_my_cpu_time();

        for(j=0; j<comm_size; j++)
        {
            MPI_Isend(send_data[j],
                      mes_length,
                      MPI_BYTE,
                      j,
                      0,
                      MPI_COMM_WORLD,
                      &send_request[j]
                     );


            MPI_Irecv(recv_data[j],
                      mes_length,
                      MPI_BYTE,
                      j,
                      0,
                      MPI_COMM_WORLD,
                      &recv_request[j]
                     );
        }



        for(j=0; j<comm_size; j++)
        {
            MPI_Waitany(comm_size,recv_request,&finished,&status);
            time_end=px_my_cpu_time();
            tmp_results[finished][i]=time_end-time_beg;
            /*
             printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
             comm_rank,comm_size,mes_length,finished,times[finished]);
            */
        }
    }

    for(i=0; i<comm_size; i++)
    {
        sum=0;
        //НОВОЕ: ПАРАЛЛЕЛЬНО ЗАПОЛНЯЕМ МАССИВ ЗАДЕРЖЕК
        for(j=0; j<num_repeats; j++)
        {
            sum+=tmp_results[i][j];
            real_times[i*num_repeats+j] = tmp_results[i][j];
            //printf("%d,%d,%d %f\n", comm_rank, i, j, tmp_results[i][j]);
        }
        times[i].average=sum/(double)num_repeats;

        st_deviation=0;
        for(j=0; j<num_repeats; j++)
        {
            st_deviation+=(tmp_results[i][j]-times[i].average)*(tmp_results[i][j]-times[i].average);
        }
        st_deviation/=(double)(num_repeats);
        times[i].deviation=sqrt(st_deviation);

        qsort(tmp_results[i], num_repeats, sizeof(px_my_time_type), clustbench_time_cmp );
        times[i].median=tmp_results[i][num_repeats/2];

        times[i].min=tmp_results[i][0];

    }

    free(send_request);
    free(recv_request);
    for(i=0; i<comm_size; i++)
    {
        if(send_data[i]!=NULL)  free(send_data[i]);
        if(recv_data[i]!=NULL)  free(recv_data[i]);
        if(tmp_results[i]!=NULL) free(tmp_results[i]);
    }
    free(send_data);
    free(recv_data);
    free(tmp_results);

    return 0;
}

int all_to_all_print_help (clustbench_benchmark_parameters_t* parameters) 
{
    printf("This is help message of all_to_all test\n");
    printf("'f' for first arg, 's' for second\n\n");
    return 0;
}

int all_to_all_define_netcdf_vars (int file_id, clustbench_benchmark_parameters_t* params) 
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

int all_to_all_put_netcdf_vars (int file_id, clustbench_benchmark_parameters_t* parameters) 
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

int all_to_all_print_parameters (clustbench_benchmark_parameters_t* parameters) 
{
    random_option_1 = (int *)parameters->benchmark_parameters;
    random_option_2 = (int *)(parameters->benchmark_parameters) + 1;
    printf("\trandom_option_1 = %d\n", *random_option_1);
    printf("\trandom_option_2 = %d\n", *random_option_2);
    return 0;
}

int all_to_all_parse_parameters (clustbench_benchmark_parameters_t* parameters,int argc,char **argv,int mpi_rank) 
{
    parameters->benchmark_parameters = (void *) malloc(sizeof(int) * 2);
    if (parameters->benchmark_parameters == NULL)
    {
        printf("Can`t allocate memory for all_to_all test individual parameters\n");
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
                    fprintf(stderr,"all_to_all_parse_parameters: Parse parameter with name 'first' failed: '%s'",
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
                    fprintf(stderr,"all_to_all_parse_parameters: Parse parameter with name 'second' failed: '%s'",
                        strerror(errno));
                }
                return 1;
            }
            break;
        case '?':
            if(!mpi_rank)
            {
                all_to_all_print_help(NULL);
            }
            return UNKNOWN_FLAG; /* flag of the unknown option */
            break;
        }
    }
    return 0;
}

int all_to_all_free_parameters (clustbench_benchmark_parameters_t* parameters)
{
    free(parameters->benchmark_parameters);
    return 0;
}
