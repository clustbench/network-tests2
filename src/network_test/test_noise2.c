/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov, Vera Y. Goritskaya 
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
 * Vera Y. Goritskaya (vera@angel.cs.msu.su)
 *
 * Ivan Beloborodov. Going to make it work.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#include "test_noise_common.h"
#include "tests_common.h"


extern test_data td;

/*
 * Test main function
 */
int test_noise(Test_time_result_type *times, int mes_length, int num_repeats, int num_noise_repeats, int loading, int num_noise_procs )
{
	int* mode_array=NULL;
	init_test_data( &td );
	
	int proc1, proc2;
	MPI_Status status;
	MPI_Request send_request;
	MPI_Request recv_request;
	
	MPI_Request* requests_noise=NULL;
	MPI_Status*  statuses_noise=NULL; 
	
	int sync_sum;

	int i, j, k, l;
	px_my_time_type time_beg,time_end;
	px_my_time_type sum;
	px_my_time_type st_deviation;
	
	int flag;
	int work_flag=1;

	int command[2];
				
	int remote_proc;

	/*
	 * Try get enough memory. If didn't, return -1. In send_request we got memory for both send and receive request
	 */ 
	requests_noise=(MPI_Request *)malloc(2*num_noise_procs*sizeof(MPI_Request));
	if(requests_noise == NULL )
	{
		return -1;
	}

	statuses_noise=(MPI_Status *)malloc(2*num_noise_procs*sizeof(MPI_Status));
	if(statuses_noise == NULL )
	{
		return -1;
	}

	mode_array=(int *)malloc(comm_size*sizeof(int));
	if(mode_array==NULL)
	{
		return -1;
	}

	if ( !alloc_test_data( &td, mes_length, num_repeats, loading, num_noise_procs ) )
	{
		return -1;
	}

	/*
	 * Ok, lets begin test part
	 */
	srand( (unsigned)time( NULL ) );
	
	for(i=0; i<comm_size; i++)
	for(j=0; j<num_repeats; j++)
	{
		td.tmp_results[i][j] = 0;
	}

	if(comm_rank==0)
	{
		/* Uncomment to debug 
		printf("HELLO! I'm 0, press any key\n");
		getchar();
		*/

		for(proc1=0;proc1<comm_size; proc1++)
		for(proc2=0;proc2<comm_size; proc2++)
		{
			flag=init_mode_array(proc1,proc2,num_noise_procs,comm_size,mode_array);
			if(flag)
			{
				return -1;
			}
			
			for(i=0;i<num_repeats;i++)
			{

				MPI_Bcast( mode_array, comm_size, MPI_INT, 0, MPI_COMM_WORLD );
				
				command[0]=i; /* Iteration number */
				command[1]=proc2; /* Leader (messages passing durations will be stored) */

				if(proc1!=0)
				{
					MPI_Send(&command,2,MPI_INT,proc1,1,MPI_COMM_WORLD);
				}
				if((proc2!=0)&&(proc2!=proc1))
				{
					MPI_Send(&command,2,MPI_INT,proc2,1,MPI_COMM_WORLD);
				}
				
				/*
				 *
				 * Goal messages in proc with number 0
				 *
				 */
				if(mode_array[0]==MODE_GOAL_MESSAGES)
				{
					if(proc1==0)
					{
						remote_proc=proc2;
					}
					else
					{
						remote_proc=proc1;
					}

					time_beg=px_my_cpu_time();

						MPI_Isend( td.send_data[remote_proc], mes_length, MPI_BYTE, remote_proc, 0, MPI_COMM_WORLD, &send_request);
						MPI_Irecv( td.recv_data[remote_proc], mes_length, MPI_BYTE, remote_proc, 0, MPI_COMM_WORLD, &recv_request);
						MPI_Wait( &send_request, &status );
						MPI_Wait( &recv_request, &status );

					time_end = px_my_cpu_time();
					
					/*
					 *
					 * command[0] -- current interation number in message passing repeats  
					 * command[1] -- process leader in the pair
					 *
					 */
					if(proc2==0)
					{
						td.tmp_results[proc1][i] = (px_my_time_type)(time_end - time_beg);
					}					
				}
				

				/*
				 *
				 * Noise messages in proc with number 0
				 *
				 */
				if(mode_array[0]==MODE_NOISE_MESSAGES)
				{
						for( j = 0; j < num_noise_repeats; j++ )
						{
							k=0;
							for( l = 1; l < comm_size; l++ )
							{
								if( mode_array[l]==MODE_NOISE_MESSAGES )
								{
									MPI_Isend( td.send_data_noise[l], loading, MPI_BYTE, l, 0, MPI_COMM_WORLD, &requests_noise[k] );
									MPI_Irecv( td.recv_data_noise[l], loading, MPI_BYTE, l, 0, MPI_COMM_WORLD, &requests_noise[k+1]);
									k+=2;
								}
						   }

						   MPI_Waitall(k,requests_noise,statuses_noise);
						}
				
				}


				/*
				 *
				 * This reduce is used for processes syncronization. All must send 
				 * their order number to the process with number 0
				 *
				 */
				MPI_Reduce(&comm_rank,&sync_sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

			} /* end for (num_repeats) */
			
		} /* End for proc1,proc2 */
		
		/*
		 *
		 * Finishing work 
		 *
		 */
		for(i=0;i<comm_size;i++)
		{
			mode_array[i]=MODE_FINISH_WORK;
		}
			
		MPI_Bcast( mode_array, comm_size, MPI_INT, 0, MPI_COMM_WORLD );
		MPI_Reduce(&comm_rank,&sync_sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);



	} /* end if(comm_rank==0) */
	else
	{
		while(work_flag)
		{
		 	MPI_Bcast( mode_array, comm_size, MPI_INT, 0, MPI_COMM_WORLD );
		 	switch(mode_array[comm_rank])
		 	{
				case MODE_GOAL_MESSAGES:
					
					MPI_Recv(&command,2,MPI_INT,0,1,MPI_COMM_WORLD,&status);
					
					remote_proc=comm_rank;					
					for(i=0;i<comm_size;i++)
					{
						if((mode_array[i]==MODE_GOAL_MESSAGES)&&(i!=comm_rank))
						{
							remote_proc=i;
							break;
						}						
					}

					time_beg=px_my_cpu_time();

						MPI_Isend( td.send_data[remote_proc], mes_length, MPI_BYTE, remote_proc, 0, MPI_COMM_WORLD, &send_request);
						MPI_Irecv( td.recv_data[remote_proc], mes_length, MPI_BYTE, remote_proc, 0, MPI_COMM_WORLD, &recv_request);
						MPI_Wait( &send_request, &status );
						MPI_Wait( &recv_request, &status );

					time_end = px_my_cpu_time();
					
					/*
					 *
					 * command[0] -- current interation number in message passing repeats  
					 * command[1] -- process leader in the pair
					 *
					 */
					if(comm_rank==command[1])
					{
						td.tmp_results[remote_proc][command[0]] = (px_my_time_type)(time_end - time_beg);
					}
					
				break;
				case MODE_NOISE_MESSAGES:
						for( i = 0; i < num_noise_repeats; i++ )
						{
							k=0;
							for( j = 0; j < comm_size; j++ )
							{
								if( (j != comm_rank) && (mode_array[j] == MODE_NOISE_MESSAGES ) )
								{
									MPI_Isend( td.send_data_noise[j], loading, MPI_BYTE, j, 0, MPI_COMM_WORLD, &requests_noise[k] );
									MPI_Irecv( td.recv_data_noise[j], loading, MPI_BYTE, j, 0, MPI_COMM_WORLD, &requests_noise[k+1]);
									k+=2;
								}
						   }

						   MPI_Waitall(k,requests_noise,statuses_noise);
						}
				
				break;
				case MODE_FINISH_WORK:
					work_flag=0;
				break;
		 	}
			
			MPI_Reduce(&comm_rank,&sync_sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

		} /* end while work_flag */ 	
	}	/* end else if(comm_rank==0) */

	/*
	 * Averaging results
	 */
	for( i = 0; i < comm_size; i++ )
	{
		sum = 0;
		for( j = 0; j < num_repeats; j++ )
		{
			sum += td.tmp_results[i][j];
		}
		times[i].average=(sum/(double)num_repeats);
 			
 		st_deviation=0;
 		for(j=0;j<num_repeats;j++)
 		{
  		 	st_deviation+=(td.tmp_results[i][j]-times[i].average)*(td.tmp_results[i][j]-times[i].average);
		}
 		st_deviation/=(double)(num_repeats);
 		times[i].deviation=sqrt(st_deviation);
 		
		/*
		 *
		 * Function my_time_cmp is described in the file  'network_test.h' and 
		 * is implemented in the file 'network_test.cpp'.
		 *
		 */
 		qsort(td.tmp_results[i], num_repeats, sizeof(px_my_time_type), my_time_cmp );
 		times[i].median=td.tmp_results[i][num_repeats/2]; 	
 		
 		times[i].min=td.tmp_results[i][0]; 	
	}

	/*
	 * Free memory
	 */
	free( requests_noise );
	free( statuses_noise );
	free( mode_array );

	free_test_data( &td );

	return 0;
}

