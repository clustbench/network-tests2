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

/*
 *****************************************************************
 *                                                               *
 * This file is one of the parus source code files. This file    *
 * written by Alexey Salnikov and will be modified by            *
 * Andreev Dmitry                                                *
 *                                                               *
 *                                                               *
 *****************************************************************
 */


#ifndef __TESTS_COMMON_H__
#define __TESTS_COMMON_H__

#include "my_time.h"
#include "types.h"


typedef struct tag_test_time_result_type
{                                       
	px_my_time_type average;
	px_my_time_type median;
	px_my_time_type deviation; 
	px_my_time_type min;
} Test_time_result_type;   

#ifdef __cplusplus
extern "C"
{
#endif

	/*
	 * This function is used for all network tests when it 
	 * counts mediane. 
	 */
	extern int my_time_cmp(const void *a, const void *b);
	extern int create_test_hosts_file
	(
		 const struct network_test_parameters_struct *parameters,
		 char **hosts_names
	);

#ifdef __cplusplus
}
#endif

/*
 * The send processor is an analog of i position
 * in marix when all coords counts in forward oderby strings
 * (0,0)->0
 * (0,1)->1
 * ...
 * (1,0)-> size
 * (1,1)-> size+1
 * ...
 * (size-1,size-1) size*size-1
 * */
#define get_send_processor(squere_coord,size) (squere_coord)/(size)
/*
 * The recv processor is an analog of j position
 * in marix when all coords counts in forward oderby strings
 * (0,0)->0
 * (0,1)->1
 * ...
 * (1,0)-> size
 * (1,1)-> size+1
 * ...
 * (size-1,size-1) size*size-1
 * */
#define get_recv_processor(squere_coord,size) (squere_coord)%(size)



#endif /* __TESTS_COMMON_H__  */


