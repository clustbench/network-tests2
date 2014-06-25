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
 * Vera Goritskaya                                               *
 * Ivan Beloborodov                                              *
 * Andreev Dmitry                                                *
 *                                                               *
 *****************************************************************
 */

#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>


#ifdef _GNU_SOURCE
#include <getopt.h>
#else
#include <unistd.h>
#endif

#include "my_time.h"
#include "my_malloc.h"
#include "parus_config.h"

#include "string_id_converters.h"
#include "data_write_operations.h"
#include "network_test2.h"
#include "types.h"
#include "easy_matrices.h"
#include "tests_common.h"
#include "parse_arguments.h"

int comm_size;
int comm_rank;



int main(int argc,char **argv)
{
    MPI_Status status;

    Test_time_result_type *times=NULL; /* old px_my_time_type *times=NULL;*/

    /*
     * The structure with network_test parameters.
     */
    struct network_test_parameters_struct test_parameters;

    /*
     * NetCDF file_id for:
     *  average
     *  median
     *  diviation
     *  minimal values
     *
     */
    int netcdf_file_av;
    int netcdf_file_me;
    int netcdf_file_di;
    int netcdf_file_mi;

    /*
     * NetCDF var_id for:
     *  average
     *  median
     *  diviation
     *  minimal values
     *
     */
    int netcdf_var_av;
    int netcdf_var_me;
    int netcdf_var_di;
    int netcdf_var_mi;

    /*
     * Variables to concentrate test results
     *
     * This is not C++ class but very like.
     */
    Easy_matrix mtr_av;
    Easy_matrix mtr_me;
    Easy_matrix mtr_di;
    Easy_matrix mtr_mi;


    char test_type_name[100];
    int i,j;


    char** host_names=NULL;
    char host_name[256];


    int flag;
	
	/*
    int help_flag = 0;
    int version_flag = 0;
	*/
    int error_flag = 0;
	/*
    int ignore_flag = 0;
    int median_flag = 0;
    int deviation_flag = 0;
    */


    int tmp_mes_size;

    /*Variables for MPI struct datatype creating*/
    MPI_Datatype struct_types[4]= {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
    MPI_Datatype MPI_My_time_struct;
    int blocklength[4]= {1,1,1,1/*,1*/};
    MPI_Aint displace[4],base;

    int step_num=0;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);

    if(comm_rank == 0)
    {
        if ( comm_size == 1 )
        {
            error_flag = 1;
            printf( "\n\nYou tries to run this programm for one MPI thread!\n\n" );
        }

        if(parse_network_test_arguments(argc,argv,&test_parameters))
        {
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }

        host_names = (char**)malloc(sizeof(char*)*comm_size);
        if(host_names==NULL)
        {
            printf("Can't allocate memory %d bytes for host_names\n",(int)(sizeof(char*)*comm_size));
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }

        for ( i = 0; i < comm_size; i++ )
        {
            host_names[i] = (char*)malloc(256*sizeof(char));
            if(host_names[i]==NULL)
            {
                printf("Can't allocate memory for name proc %d\n",i);
                MPI_Abort(MPI_COMM_WORLD,-1);
            }
        }
    } /* End if(rank==0) */

    /*
     * Going to get and write all processors' hostnames
     */
    gethostname( host_name, 255 );

    if ( comm_rank == 0 )
    {
        for ( i = 1; i < comm_size; i++ )
            MPI_Recv( host_names[i], 256, MPI_CHAR, i, 200, MPI_COMM_WORLD, &status );
        strcpy(host_names[0],host_name);
    }
    else
    {
        MPI_Send( host_name, 256, MPI_CHAR, 0, 200, MPI_COMM_WORLD );
    }

    /*
     * Initializing num_procs parameter
     */
    test_parameters.num_procs=comm_size;

    if( comm_rank == 0)
    {
        /*
         *
         * Matrices initialization
         *
         */
        flag = easy_mtr_create(&mtr_av,comm_size,comm_size);
        if( flag==-1 )
        {
            printf("Can not to create average matrix to story the test results\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
        flag = easy_mtr_create(&mtr_me,comm_size,comm_size);
        if( flag==-1 )
        {
            printf("Can not to create median matrix to story the test results\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
        flag = easy_mtr_create(&mtr_di,comm_size,comm_size);
        if( flag==-1 )
        {
            printf("Can not to create deviation matrix to story the test results\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
        flag = easy_mtr_create(&mtr_mi,comm_size,comm_size);
        if( flag==-1 )
        {
            printf("Can not to create min values matrix to story  the test results\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }

        if(create_netcdf_header(AVERAGE_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_av,&netcdf_var_av))
        {
            printf("Can not to create file with name \"%s_average.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }

        if(create_netcdf_header(MEDIAN_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_me,&netcdf_var_me))
        {
            printf("Can not to create file with name \"%s_median.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }

        if(create_netcdf_header(DEVIATION_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_di,&netcdf_var_di))
        {
            printf("Can not to create file with name \"%s_deviation.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }

        if(create_netcdf_header(MIN_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_mi,&netcdf_var_mi))
        {
            printf("Can not to create file with name \"%s_min.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }

	if(create_test_hosts_file(&test_parameters,host_names))		
	{
		printf("Can not to create file with name \"%s_hosts.txt\"\n",test_parameters.file_name_prefix);
    		MPI_Abort(MPI_COMM_WORLD,-1);
    		return -1;
	}

        /*
         *
         * Printing initial message for user
         *
         */
        printf("network test (%d processes):\n\n", comm_size);
        get_test_type_name(test_parameters.test_type,test_type_name);
        printf("\ttest type\t\t\t\"%s\"\n",test_type_name);
        printf("\tbegin message length\t\t%d\n",test_parameters.begin_message_length);
        printf("\tend message length\t\t%d\n",test_parameters.end_message_length);
        printf("\tstep length\t\t\t%d\n",test_parameters.step_length);
        printf("\tnoise message length\t\t%d\n",test_parameters.noise_message_length);
        printf("\tnumber of noise messages\t%d\n",test_parameters.num_noise_messages);
        printf("\tnumber of noise processes\t%d\n",test_parameters.num_noise_procs);
        printf("\tnumber of repeates\t\t%d\n",test_parameters.num_repeats);
        printf("\tresult file average\t\t\"%s_average.nc\"\n",test_parameters.file_name_prefix);
        printf("\tresult file median\t\t\"%s_median.nc\"\n",test_parameters.file_name_prefix);
        printf("\tresult file deviation\t\t\"%s_deviation.nc\"\n",test_parameters.file_name_prefix);
        printf("\tresult file minimum\t\t\"%s_min.nc\"\n",test_parameters.file_name_prefix);
	printf("\tresult file hosts\t\t\"%s_hosts.txt\"\n\n",test_parameters.file_name_prefix);


    } /* End preparation (only in MPI process with rank 0) */

    /*
     * Broadcasting command line parametrs
     *
     * The structure network_test_parameters_struct contains 9
     * parametes those are placed at begin of structure.
     * So, we capable to think that it is an array on integers.
     *
     * Little hack from Alexey Salnikov.
     */
    MPI_Bcast(&test_parameters,9,MPI_INT,0,MPI_COMM_WORLD);


    /*
     * Creating struct time type for MPI operations
     */
    {
        Test_time_result_type tmp_time;
        MPI_Address( &(tmp_time.average), &base);
        MPI_Address( &(tmp_time.median), &displace[1]);
        MPI_Address( &(tmp_time.deviation), &displace[2]);
        MPI_Address( &(tmp_time.min), &displace[3]);
    }
    displace[0]=0;
    displace[1]-=base;
    displace[2]-=base;
    displace[3]-=base;
    MPI_Type_struct(4,blocklength,displace,struct_types,&MPI_My_time_struct);
    MPI_Type_commit(&MPI_My_time_struct);


    times=(Test_time_result_type* )malloc(comm_size*sizeof(Test_time_result_type));
    if(times==NULL)
    {
		printf("Memory allocation error\n");
		MPI_Abort(MPI_COMM_WORLD,-1);
		return -1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    

    /*
     * Circle by length of messages
     */
    for
	    (
	     tmp_mes_size=test_parameters.begin_message_length;
	     tmp_mes_size<test_parameters.end_message_length;
	     step_num++,tmp_mes_size+=test_parameters.step_length
	     )
    {
        if(test_parameters.test_type==ALL_TO_ALL_TEST_TYPE)
        {
            all_to_all(times,tmp_mes_size,test_parameters.num_repeats);
        }
        if(test_parameters.test_type==BCAST_TEST_TYPE)
        {
            bcast(times,tmp_mes_size,test_parameters.num_repeats);
        }

        if(test_parameters.test_type==NOISE_BLOCKING_TEST_TYPE)
        {
                test_noise_blocking
		(
		 	times,
		    	tmp_mes_size, 
			test_parameters.num_repeats, 
			test_parameters.num_noise_messages, 
			test_parameters.noise_message_length,
		       	test_parameters.num_noise_procs
		);
        }

        if(test_parameters.test_type==NOISE_TEST_TYPE)
        {
            		test_noise
			(
			 	times,
				tmp_mes_size, 
				test_parameters.num_repeats, 
				test_parameters.num_noise_messages, 
				test_parameters.noise_message_length,
				test_parameters.num_noise_procs
			);
        }

        if(test_parameters.test_type==ONE_TO_ONE_TEST_TYPE)
        {
            one_to_one(times,tmp_mes_size,test_parameters.num_repeats);
        } /* end one_to_one */

        if(test_parameters.test_type==ASYNC_ONE_TO_ONE_TEST_TYPE)
        {
            async_one_to_one(times,tmp_mes_size,test_parameters.num_repeats);
        } /* end async_one_to_one */

        if(test_parameters.test_type==SEND_RECV_AND_RECV_SEND_TEST_TYPE)
        {
            send_recv_and_recv_send(times,tmp_mes_size,test_parameters.num_repeats);
        } /* end send_recv_and_recv_send */




        if(test_parameters.test_type==PUT_ONE_TO_ONE_TEST_TYPE)
        {
		put_one_to_one(times,tmp_mes_size,test_parameters.num_repeats);
        } /* end put_one_to_one */

        if(test_parameters.test_type==GET_ONE_TO_ONE_TEST_TYPE)
        {
		get_one_to_one(times,tmp_mes_size,test_parameters.num_repeats);
        } /* end get_one_to_one */


        MPI_Barrier(MPI_COMM_WORLD);

        if(comm_rank==0)
        {
            for(j=0; j<comm_size; j++)
            {
                MATRIX_FILL_ELEMENT(mtr_av,0,j,times[j].average);
                MATRIX_FILL_ELEMENT(mtr_me,0,j,times[j].median);
                MATRIX_FILL_ELEMENT(mtr_di,0,j,times[j].deviation);
                MATRIX_FILL_ELEMENT(mtr_mi,0,j,times[j].min);
            }
            for(i=1; i<comm_size; i++)
            {

                MPI_Recv(times,comm_size,MPI_My_time_struct,i,100,MPI_COMM_WORLD,&status);
                for(j=0; j<comm_size; j++)
                {
                    MATRIX_FILL_ELEMENT(mtr_av,i,j,times[j].average);
                    MATRIX_FILL_ELEMENT(mtr_me,i,j,times[j].median);
                    MATRIX_FILL_ELEMENT(mtr_di,i,j,times[j].deviation);
                    MATRIX_FILL_ELEMENT(mtr_mi,i,j,times[j].min);

                }
            }


            if(netcdf_write_matrix(netcdf_file_av,netcdf_var_av,step_num,mtr_av.sizex,mtr_av.sizey,mtr_av.body))
            {
                printf("Can't write average matrix to file.\n");
                MPI_Abort(MPI_COMM_WORLD,-1);
                return 1;
            }

            if(netcdf_write_matrix(netcdf_file_me,netcdf_var_me,step_num,mtr_me.sizex,mtr_me.sizey,mtr_me.body))
            {
                printf("Can't write median matrix to file.\n");
                MPI_Abort(MPI_COMM_WORLD,-1);
                return 1;
            }

            if(netcdf_write_matrix(netcdf_file_di,netcdf_var_di,step_num,mtr_di.sizex,mtr_di.sizey,mtr_di.body))
            {
                printf("Can't write deviation matrix to file.\n");
                MPI_Abort(MPI_COMM_WORLD,-1);
                return 1;
            }

            if(netcdf_write_matrix(netcdf_file_mi,netcdf_var_mi,step_num,mtr_mi.sizex,mtr_mi.sizey,mtr_mi.body))
            {
                printf("Can't write  matrix with minimal values to file.\n");
                MPI_Abort(MPI_COMM_WORLD,-1);
                return 1;
            }


            printf("message length %d finished\r",tmp_mes_size);
            fflush(stdout);

        } /* end comm rank 0 */
        else
        {
            MPI_Send(times,comm_size,MPI_My_time_struct,0,100,MPI_COMM_WORLD);
        }

      
        /* end for cycle .
         * Now we  go to the next length of message that is used in
         * the test perfomed on multiprocessor.
         */
    }
    
    /* TODO
     * Now free times array.
     * It should be changed in future for memory be allocated only once.
     *
     * Times array should be moved from return value to the input argument
     * for any network_test.
     */
    
	free(times);


	if(comm_rank==0)
    {

        netcdf_close_file(netcdf_file_av);
        netcdf_close_file(netcdf_file_me);
        netcdf_close_file(netcdf_file_di);
        netcdf_close_file(netcdf_file_mi);

        for(i=0; i<comm_size; i++)
        {
            free(host_names[i]);
        }
        free(host_names);

        free(mtr_av.body);
        free(mtr_me.body);
        free(mtr_di.body);
        free(mtr_mi.body);

        printf("\nTest is done\n");
    }

    MPI_Finalize();
    return 0;
} /* main finished */

