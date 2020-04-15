
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <mpi.h>

#include "my_time.h"
#include "my_malloc.h"
#include "tests_common.h"
#include "data_write_operations.h"

extern int comm_rank;
extern int comm_size;

double* real_directed_one_to_one(int mes_length,int num_repeats,int source_proc,int dest_proc);

int directed_one_to_one(equality_class* eq_classes, int total_classes, struct network_test_parameters_struct* params)
{
    int pair[3];
    int beg_mes_length;
    int end_mes_length;
    int step;
    int num_repeats = params->num_repeats;
    int q, mes_len, i;
    int confirmation_flag;

    int send_proc,recv_proc;

    MPI_Status status;

    double *classes_times;

    double *recv_times;
    classes_times = (double*)malloc(sizeof(double) * comm_size * comm_size * num_repeats);
    
    
    if(comm_rank == 0)
    {
        for(q = 0; q < total_classes; ++q)
        {
            char filename[255];
            memset(filename, 0, 255);
            sprintf(filename, "%d_class.nc", q);
            int netcdf_class_file;
            int netcdf_repeats_data;

            create_netcdf_header4d_eq_class(params, &netcdf_class_file, &netcdf_repeats_data, filename);

            beg_mes_length = params->begin_message_length;
            end_mes_length = params->end_message_length;
            step = params->step_length;
            for (mes_len = beg_mes_length; mes_len < end_mes_length; mes_len += step)
            {
                memset(classes_times, -1.0,  comm_size * comm_size * num_repeats * sizeof(double));
                for(i = 0; i < eq_classes[q].links_count; ++i)
                {
                    send_proc=eq_classes[q].links[i].send_rank;
                    recv_proc=eq_classes[q].links[i].recv_rank;

                    pair[0]=send_proc;
                    pair[1]=recv_proc;
                    pair[2]=mes_len;


                    if(send_proc)
                        MPI_Send(pair,3,MPI_INT,send_proc,1,MPI_COMM_WORLD);
                    if(recv_proc)
                        MPI_Send(pair,3,MPI_INT,recv_proc,1,MPI_COMM_WORLD);

                    if(recv_proc == 0)
                    {
                        recv_times=real_directed_one_to_one(mes_len,num_repeats,send_proc,recv_proc);
                    }
                    if(send_proc == 0)
                    {
                        real_directed_one_to_one(mes_len,num_repeats,send_proc,recv_proc);
                    }
                    if(send_proc)
                    {
                        MPI_Recv(&confirmation_flag,1,MPI_INT,send_proc,1,MPI_COMM_WORLD,&status);
                    }

                    if(recv_proc)
                    {
                        recv_times = (double*)malloc(sizeof(double) * num_repeats);
                        MPI_Recv( recv_times, num_repeats, MPI_DOUBLE, recv_proc, 1, MPI_COMM_WORLD, &status);
                        MPI_Recv(&confirmation_flag,1,MPI_INT,recv_proc,1,MPI_COMM_WORLD,&status);
                    }
                    double *class_beg_cpy = classes_times + send_proc * comm_size * num_repeats + recv_proc * num_repeats;
                    memcpy(class_beg_cpy, recv_times, num_repeats);
                    free(recv_times);
                } 
                printf("Message length for class %d: %d is done\n", q, mes_len);
                netcdf_write_matrix4d(netcdf_class_file, netcdf_repeats_data, mes_len / step,comm_size, comm_size,num_repeats, classes_times);
            }
            //save to file
            netcdf_close_file(netcdf_class_file);
        }
            free(classes_times);
            pair[0] = -1;
            for(i = 1; i < comm_size; i++)
                MPI_Send(pair,3,MPI_INT,i,1,MPI_COMM_WORLD);
    }
    else
    {
        while(1)
        {
            MPI_Recv(pair,3,MPI_INT,0,1,MPI_COMM_WORLD,&status);
            send_proc=pair[0];
            recv_proc=pair[1];
            int mes_len=pair[2];

	    if(send_proc==-1)
                break;
            if(send_proc==comm_rank)
                real_directed_one_to_one(mes_len,num_repeats,send_proc,recv_proc);
            if(recv_proc==comm_rank)
            {
                recv_times=real_directed_one_to_one(mes_len,num_repeats,send_proc,recv_proc);
                MPI_Send(recv_times,num_repeats,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
                free(recv_times);
            }
            confirmation_flag=1;
            MPI_Send(&confirmation_flag,1,MPI_INT,0,1,MPI_COMM_WORLD);
        }
    } /* end else comm_rank==0 */
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
} /* end one_to_one */


double* real_directed_one_to_one(int mes_length,int num_repeats,int source_proc,int dest_proc)
{
    double time_beg,time_end;
    char *data=NULL;
    MPI_Status status;
    int i;
    double* iter_times = NULL;

    iter_times = (double*)malloc(sizeof(double) * num_repeats);

    int tmp;

    if(source_proc==dest_proc)
    {
        memset(iter_times, 0, sizeof(double) * num_repeats);
        return iter_times;
    }

    if(iter_times==NULL)
    {
        printf("proc %d from %d: Can not allocate memory\n",comm_rank,comm_size);
        return iter_times;
    }

    data=(char *)malloc(mes_length*sizeof(char));
    if(data==NULL)
    {
        free(iter_times);
        printf("proc %d from %d: Can not allocate memory\n",comm_rank,comm_size);
        return iter_times;
    }



    for(i=0; i<num_repeats; i++)
    {

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

            iter_times[i]=(time_end-time_beg);
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
            iter_times[i]=(time_end-time_beg);

            MPI_Send(&comm_rank,1,MPI_INT,source_proc,100,MPI_COMM_WORLD);
            /*
             printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
             comm_rank,comm_size,mes_length,finished,times[finished]);
            */
        }
    }

        return iter_times;

}

