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

#include <stdlib.h>
#include <string.h>

//#include <netcdfcpp.h>

#include "my_time.h"
#include "string_id_converters.h"

#include "network_speed.h"

#include "types.h"
#include "data_write_operations.h"
#include "tests_common.h"

#define BUFFSIZE 256
#define CHECKSIZE 15
#define TEST_STRING "Message length"


int main(int argc, char** argv)
{
   struct network_test_parameters_struct test_parameters;
   
    if (argc < 3)
    {
        printf("Use: convert_to_netcdf <inputfile_path> <outputfile_path>");
    }

    Network_speed test_data;

    char* file_name = argv[1];
    int netcdf_file;
    int netcdf_var;

    
    test_data.fread(file_name);

    if ( test_data.is_no_file( ) || test_data.is_processor_info( ) )
    {
        printf( "Bad file. No test parameters info.\n" );
	    return 1;
    }


    if (! test_data.is_test_info( ) )
    {
        printf("Sorry, information on test parameters is epsent in file '%s'",argv[1]);
        return 1;
    }
        /*
         * Building test parameters
         */
        test_parameters.num_procs = test_data.get_num_processors( );
        test_parameters.begin_message_length = test_data.get_message_begin_length( );
        test_parameters.end_message_length = test_data.get_message_end_length( );
        test_parameters.step_length = test_data.get_step_length( );
        test_parameters.noise_message_length = test_data.get_noise_message_length( );
        test_parameters.num_noise_messages = test_data.get_noise_message_num( );
        test_parameters.num_noise_procs = test_data.get_number_of_noise_processors( );
        test_parameters.num_repeats = test_data.get_number_of_repeates( );
        test_parameters.file_name_prefix=argv[2];

        char str[BUFFSIZE];
        test_data.get_test_type( str );
        int test_type = get_test_type(str);
        int data_type = 0;

        if(create_netcdf_header(test_type,&test_parameters,&netcdf_file,&netcdf_var))
        {
            printf("Can not to create NetCDF file with prefix \"%s\"\n",test_parameters.file_name_prefix);
            return 1;
        }

        if(create_test_hosts_file(&test_parameters,test_data.get_host_names()))
        {
             printf("Can not to create file with name \"%s_hosts.txt\"\n",test_parameters.file_name_prefix);                
             return 1;
        }

        for(int i=0;i<test_data.get_num_messages();i++)
        {
             if(netcdf_write_matrix(
                                        netcdf_file,
                                        netcdf_var,
                                        i,
                                        test_parameters.num_procs,
                                        test_parameters.num_procs,
                                        test_data.get_certain_matix(i).get_body()
                                    )
                ) /* End condition */
             {
                printf("Sorry. write matrix to netCDF filed\n");
                return 1;
             }             
        }
    
        netcdf_close_file(netcdf_file);
        printf("Done\n");

        return 0;
}

