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
#include <mpi.h>

#include "my_time.h"
#include "my_malloc.h"
#include "tests_common.h"


extern int comm_rank;
extern int comm_size;

Test_time_result_type real_one_to_one(int mes_length,int num_repeats,int source_proc,int dest_proc);

int one_to_one(Test_time_result_type *times,int mes_length,int num_repeats)
{
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
                times[send_proc]=real_one_to_one(mes_length,num_repeats,send_proc,recv_proc);
            }
            if(send_proc==0)
            {
                real_one_to_one(mes_length,num_repeats,send_proc,recv_proc);
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
                real_one_to_one(mes_length,num_repeats,send_proc,recv_proc);
            if(recv_proc==comm_rank)
                times[send_proc]=real_one_to_one(mes_length,num_repeats,send_proc,recv_proc);

            confirmation_flag=1;
            MPI_Send(&confirmation_flag,1,MPI_INT,0,1,MPI_COMM_WORLD);
        }
    } /* end else comm_rank==0 */

    return 0;
} /* end one_to_one */


Test_time_result_type real_one_to_one(int mes_length,int num_repeats,int source_proc,int dest_proc)
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
    }
    times.average=(sum/(double)num_repeats);

    st_deviation=0;
    for(i=0; i<num_repeats; i++)
    {
        st_deviation+=(tmp_results[i]-times.average)*(tmp_results[i]-times.average);
    }
    st_deviation/=(double)(num_repeats);
    times.deviation=sqrt(st_deviation);

    qsort(tmp_results, num_repeats, sizeof(px_my_time_type), my_time_cmp );
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

