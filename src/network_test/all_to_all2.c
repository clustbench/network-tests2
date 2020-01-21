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

#include "my_time.h"
#include "my_malloc.h"
#include "tests_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

extern int comm_rank;
extern int comm_size;

int all_to_all(Test_time_result_type *times,int mes_length,int num_repeats);

int all_to_all(Test_time_result_type *times,int mes_length,int num_repeats)
{
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
        return -1;
    }

    send_request=(MPI_Request *)malloc(comm_size*sizeof(MPI_Request));
    if(send_request == NULL)
    {
        free(times);
        free(tmp_results);
        return -1;
    }

    recv_request=(MPI_Request *)malloc(comm_size*sizeof(MPI_Request));
    if(recv_request == NULL)
    {
        free(times);
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
        for(j=0; j<num_repeats; j++)
        {
            sum+=tmp_results[i][j];
        }
        times[i].average=sum/(double)num_repeats;

        st_deviation=0;
        for(j=0; j<num_repeats; j++)
        {
            st_deviation+=(tmp_results[i][j]-times[i].average)*(tmp_results[i][j]-times[i].average);
        }
        st_deviation/=(double)(num_repeats);
        times[i].deviation=sqrt(st_deviation);

        qsort(tmp_results[i], num_repeats, sizeof(px_my_time_type), my_time_cmp );
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
