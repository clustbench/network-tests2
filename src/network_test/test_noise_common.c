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
 * Ivan Beloborodov
 *
 */

#include "my_time.h"
/*#include "line_dynamic_array.h" 
#include "id.h"
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define __TEST_NOISE_COMMON_C__
#include "test_noise_common.h"
#undef __TEST_NOISE_COMMON_C__



test_data td;

/**
 * Initialization of test data
 */
void init_test_data( test_data* td )
{
	td->tmp_results = NULL;
	td->send_data = NULL;
	td->recv_data = NULL;
	td->send_data_noise = NULL;
	td->recv_data_noise = NULL;
	td->processors = NULL;
}

/**
 * Clear memory if one of allocation wasn't sucessful.
 * Consider allocation is made sequentially
 * Free memory block til we met one not supposed for freeing
 */
void clear_test_data( test_data* td )
{
	if ( td->tmp_results ) free( td->tmp_results ); else return;
	if ( td->send_data ) free( td->send_data ); else return;
	if ( td->recv_data ) free( td->recv_data ); else return;
	if ( td->send_data_noise ) free( td->send_data_noise ); else return;
	if ( td->recv_data_noise ) free( td->recv_data_noise ); else return;
	if ( td->processors ) free( td->processors ); else return;
}

/**
 * The same for sub-arrays of send-receive data
 */
void clear_more_test_data( test_data* td, int i )
{
	int j;

	if ( td->tmp_results[i] )
	{
		free( td->tmp_results[i] );
		if ( td->send_data[i] )
		{
			free( td->send_data[i] );
			if ( td->recv_data[i] )
			{
				free( td->recv_data[i] );
				if ( td->recv_data_noise[i] )
				{
					free( td->recv_data_noise[i] );
					if ( td->send_data_noise[i] )
						free( td->send_data_noise[i] );
				}
			}
		}
	}

	for ( j = i - 1; j >= 0; j-- )
	{
		free( td->tmp_results[i] );
		free( td->send_data[i] );
		free( td->recv_data[i] );
		free( td->send_data_noise[i] );
		free( td->recv_data_noise[i] );
	}
}

/**				
 * Try to allocate memory for all test data. If unsucessful,
 * clear already allocated and return 0
 */
int alloc_test_data( test_data* td, int mes_length, int num_repeats, int loading, int num_processors )
{
 
	if ( !( td->tmp_results = (px_my_time_type**)
		malloc( comm_size * sizeof(px_my_time_type*) ) ) )
	{
		clear_test_data( td );
		return 0;
	}

	if ( !( td->send_data = (char**)
		malloc( sizeof(char*) * comm_size ) ) )
	{
		clear_test_data( td );
		return 0;
	}

	if ( !( td->recv_data = (char**)
		malloc( sizeof(char*) * comm_size ) ) )
	{
		clear_test_data( td );
		return 0;
	}

	if ( !( td->send_data_noise = (char**)
		malloc( sizeof(char*) * comm_size ) ) )
	{
		clear_test_data( td );
		return 0;
	}

	if ( !( td->recv_data_noise = (char**)
		malloc( sizeof(char*) * comm_size ) ) )
	{
		clear_test_data( td );
		return 0;
	}

	if ( !( td->processors = (int*)
		malloc( sizeof(int) * num_processors ) ) )
	{
		clear_test_data( td );
		return 0;
	}

	int i;

	for( i = 0; i < comm_size; i++)
	{
		td->tmp_results[i] = NULL;
		td->send_data[i] = NULL;
		td->recv_data[i] = NULL;
		td->send_data_noise[i] = NULL;
		td->recv_data_noise[i] = NULL;

		if ( !( td->tmp_results[i] = (px_my_time_type*)
			malloc( num_repeats * sizeof( px_my_time_type ) ) ) )
		{
			clear_more_test_data( td, i );
			return 0;
		}

		if ( !( td->send_data[i] = (char*)
			malloc( mes_length * sizeof(char) ) ) )
		{
			clear_more_test_data( td, i );
			return 0;
		}

		if( !( td->recv_data[i] = (char*)
			malloc( mes_length * sizeof(char) ) ) )
		{
			clear_more_test_data( td, i );
			return 0;
		}

		if ( !( td->send_data_noise[i] = (char*)
			malloc( loading * sizeof(char) ) ) )
		{
			clear_more_test_data( td, i );
			return 0;
		}

		if ( !( td->recv_data_noise[i] = (char*)
			malloc( loading * sizeof(char) ) ) )
		{
			clear_more_test_data( td, i );
			return 0;
		}
	}

	return 1;
}

void free_test_data( test_data* td )
{
	free( td->processors );
	
	int i;
	
	for ( i = 0; i < comm_size; i++)
	{
		free( td->tmp_results[i] );
		free( td->send_data[i] );
		free( td->recv_data[i] );
		free( td->send_data_noise[i] );
		free( td->recv_data_noise[i] );
	}
	
	free( td->tmp_results );
	free( td->send_data );
	free( td->recv_data );
	free( td->send_data_noise );
	free( td->recv_data_noise );
}

/**
 * Going to select randomly num_processors processors from all, excluding proc1 and proc2
 */
/*
int random_choice( int proc1, int proc2, int num_processors, int* processors )
{

	int i;
	int flag;
	ID *id;

	int number_in_dynamic_array;
	
	Line_dynamic_array<ID> free_processors;

	/*
	 * Check for boundaries
	 * /
	if( 
		( comm_size <= 2 ) ||
		( num_processors <= 0 ) ||
		( num_processors > comm_size - 2 ) 
	  )
	{
		return 0;
	}

	for(i=0;i<comm_size;i++)
	{
		if((i!=proc1)&&(i!=proc2))
		{
			ID id1=i;
			flag=free_processors.add_element(&id1);
			if(flag)
			{
				return 0;
			}
		}
	}

	for(i=0;i<num_processors;i++)
	{
		number_in_dynamic_array=(int)(((double)rand()/RAND_MAX)*(free_processors.num_elements()-1));
		id=free_processors.look_position_uncopy(number_in_dynamic_array);
		processors[i]=*id;
		free_processors.delete_element(number_in_dynamic_array);
	}

	/*
	 r = (int)( rand( ) * double( comm_size - 2 - i ) / ( RAND_MAX + 1. ) );
	 * /
	return 1;
}
*/
/*
int init_mode_array(int proc1,int proc2,int num_noise_procs,int num_all_procs,int *mode_array)
{
	int i;
	int *noise_procs=NULL;

	noise_procs=(int *)malloc(num_noise_procs*sizeof(int));
	if(noise_procs==NULL)
	{
		return -1;
	}
	
	for(i=0;i<num_all_procs;i++)
	{
		mode_array[i]=MODE_IDLE;
	}

	mode_array[proc1]=MODE_GOAL_MESSAGES;
	mode_array[proc2]=MODE_GOAL_MESSAGES;

	

	if(!random_choice(proc1,proc2, num_noise_procs,noise_procs))
	{
		return -2;
	}
	
	for(i=0;i<num_noise_procs;i++)
	{
		mode_array[noise_procs[i]]=MODE_NOISE_MESSAGES;
	}

	free(noise_procs);

	return 0;
}
*/

int init_mode_array(int proc1,int proc2,int num_noise_procs,int num_all_procs,int *mode_array)
{	
	// is num_all_procs==comm_size???? WTF??
	int i;
		
	for(i=0;i<num_all_procs;i++)
	{
		mode_array[i]=MODE_IDLE;
	}

	mode_array[proc1]=MODE_GOAL_MESSAGES;
	mode_array[proc2]=MODE_GOAL_MESSAGES;

	/*

	if(!random_choice(proc1,proc2, num_noise_procs,noise_procs))
	{
		return -2;
	}
	
	for(i=0;i<num_noise_procs;i++)
	{
		mode_array[noise_procs[i]]=MODE_NOISE_MESSAGES;
	}
	*/
	
	if( 
		( num_all_procs <= 2 ) ||
		( num_noise_procs <= 0 ) ||
		( num_noise_procs > num_all_procs - 2 ) 
	  )
	{
		return 0;
	}

	int r=0;
	int k=(num_all_procs-2)/num_noise_procs;
	for ( i = 0; i < num_noise_procs; i++ )
	{
		while((r==proc1)||(r==proc2))
			r++;
		mode_array[r]=MODE_NOISE_MESSAGES;
		r+=k;
	}

	
	return 0;
}
