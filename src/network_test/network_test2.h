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


#ifndef __NETWORK_TEST2_H__
#define __NETWORK_TEST2_H__


#include "my_time.h"
#include "tests_common.h"

#define UNKNOWN_TEST_TYPE 0
#define ONE_TO_ONE_TEST_TYPE 1
#define ALL_TO_ALL_TEST_TYPE 2
#define ASYNC_ONE_TO_ONE_TEST_TYPE 3
#define SEND_RECV_AND_RECV_SEND_TEST_TYPE 4
#define NOISE_TEST_TYPE 5
#define NOISE_BLOCKING_TEST_TYPE 6
#define BCAST_TEST_TYPE 7
#define PUT_ONE_TO_ONE_TEST_TYPE 8
#define GET_ONE_TO_ONE_TEST_TYPE 9

#ifdef __cplusplus
extern "C"
{
#endif

extern int all_to_all(Test_time_result_type *times,int mes_length,int num_repeats);
extern int async_one_to_one(Test_time_result_type *times,int mes_length,int num_repeats);
extern int bcast(Test_time_result_type *times,int mes_length,int num_repeats);
extern int one_to_one(Test_time_result_type *times,int mes_length,int num_repeats);
extern int send_recv_and_recv_send(Test_time_result_type *times,int mes_length,int num_repeats);
extern int test_noise(Test_time_result_type *times,int mes_length, int num_repeats, int num_noise_repeats, int noise_message_length, int num_noise_procs);
extern int test_noise_blocking(Test_time_result_type *times,int mes_length, int num_repeats, int num_noise_repeats, int noise_message_length, int num_noise_procs);
extern int get_one_to_one(Test_time_result_type *times,int mes_length,int num_repeats);
extern int put_one_to_one(Test_time_result_type *times,int mes_length,int num_repeats);


#ifdef __cplusplus
}
#endif

#endif /* __NETWORK_TEST2_H__ */

