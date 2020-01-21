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

Test_time_result_type *bcast(Test_time_result_type *times, int mes_length, int num_repeats);

Test_time_result_type *bcast(Test_time_result_type *times, int mes_length, int num_repeats)
{
    px_my_time_type **tmp_results=NULL;
    px_my_time_type time_beg,time_end;
    char *data=NULL;
    px_my_time_type st_deviation;
    int i,j;
    int flag=0;
    double sum;


    tmp_results=(px_my_time_type**)malloc(comm_size*sizeof(px_my_time_type*));
    if(tmp_results==NULL)
    {
        free(times);
        return NULL;
    }

    data=(char *)malloc(mes_length*sizeof(char));
    if(data == NULL)
    {
        free(times);
        free(tmp_results);
        return NULL;
    }

    for(i=0; i<comm_size; i++)
    {
        tmp_results[i]=NULL;

        tmp_results[i]=(px_my_time_type *)malloc(num_repeats*sizeof(px_my_time_type));
        if(tmp_results[i]==NULL)
        {
            flag=1;
        }
    }

    if(flag == 1)
    {
        free(times);
        free(data);
        for(i=0; i<comm_size; i++)
        {
            if(tmp_results[i]!=NULL) free(tmp_results[i]);
        }
        free(tmp_results);
        return NULL;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for(j=0; j<comm_size; j++)
    {
        for(i=0; i<num_repeats; i++)
        {
            time_beg=px_my_cpu_time();
            MPI_Bcast(data,
                      mes_length,
                      MPI_BYTE,
                      j,
                      MPI_COMM_WORLD
                     );
            time_end=px_my_cpu_time();
            tmp_results[j][i]=time_end-time_beg;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        /*
         for(j=0;j<comm_size;j++)
         {
          MPI_Waitany(comm_size,recv_request,&finished,&status);
          time_end=px_my_cpu_time();
          tmp_results[j][i]=time_end-time_beg;

          // printf("process %d from %d:\n  Finished recive message length=%d from %d throug the time %ld\n",
           //comm_rank,comm_size,mes_length,finished,times[finished]);

         }*/
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
    for(i=0; i<comm_size; i++)
    {
        if(tmp_results[i]!=NULL) free(tmp_results[i]);
    }
    free(data);
    free(tmp_results);

    return times;
}
