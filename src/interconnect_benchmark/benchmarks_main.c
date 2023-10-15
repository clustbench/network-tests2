/*
 * This file is a part of the Clustbench project.
 * 
 * There is entry point for all benchmarks.  
 *  
 *
 * Alexey N. Salnikov (salnikov@cs.msu.ru)
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
#include <stdio.h>
#include <stdint.h>


//зачем это? для работы с графической оболочкой?
#ifdef _GNU_SOURCE
#include <getopt.h>
#else
#include <unistd.h>
#endif

/*
from ../..
*/
//это подключает стандартный конфиг?
#include "clustbench_config.h"

/*
from ../core 
*/
#include "string_id_converters.h"
#include "my_time.h"

/*
from common
*/
#include "clustbench_time.h"
#include "clustbench_malloc.h"
#include "clustbench_types.h"
#include "clustbench_easy_matrices.h"
#include "clustbench_plugin_operations.h"
#include "clustbench_data_write_operations.h"
#include "benchmarks_common.h"

/*
from ./
*/
#include "get_node_name.h"
#include "parse_arguments.h"

int comm_size;
int comm_rank;

int main(int argc,char **argv)
{
    MPI_Status status;

    clustbench_time_result_t *times = NULL; /* old px_my_time_type *times=NULL;*/
    clustbench_time_t *real_times = NULL;

    /*
     * The structure with network_test parameters.
     */
    clustbench_benchmark_parameters_t test_parameters;

    /*
     * NetCDF file_id for:
     *  average
     *  median
     *  diviation
     *  minimal values
     *  all values
     *
     */
     
    //добавил переменные под матрицу с массивами задержек (с постфиксом del)
    int netcdf_file_av;
    int netcdf_file_me;
    int netcdf_file_di;
    int netcdf_file_mi;
    int netcdf_file_all;

    /*
     * NetCDF var_id for:
     *  average
     *  median
     *  diviation
     *  minimal values
     *  all values
     *
     */
    int netcdf_var_av;
    int netcdf_var_me;
    int netcdf_var_di;
    int netcdf_var_mi;
    int netcdf_var_all;

    /*
     * Variables to concentrate test results
     *
     * This is not C++ class but very like.
     */
    Clustbench_easy_matrix     mtr_av;
    Clustbench_easy_matrix     mtr_me;
    Clustbench_easy_matrix     mtr_di;
    Clustbench_easy_matrix     mtr_mi;
    Clustbench_easy_matrix_3d  mtr_all;


    uint32_t i,j;


    char** host_names=NULL;
    char host_name[CLUSTBENCH_HOSTNAME_LENGTH];


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
    MPI_Datatype struct_types[4]= {
                                    CLUSTBENCH_MPI_TIME_T,
                                    CLUSTBENCH_MPI_TIME_T,
                                    CLUSTBENCH_MPI_TIME_T,
                                    CLUSTBENCH_MPI_TIME_T
                                  };

    MPI_Datatype MPI_My_time_struct;
    int blocklength[4]= {1,1,1,1/*,1*/};
    MPI_Aint displace[4],base;

    int step_num = 0;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);

    /*
     * Initializing num_procs parameter
     */
    error_flag = parse_network_test_arguments(&test_parameters,argc,argv,comm_rank);
    if(error_flag == ERROR_FLAG)
    {
            MPI_Abort(MPI_COMM_WORLD,1);
            return 1;
    }

    if(error_flag == LIST_FLAG)
    {
        if(comm_rank == 0)
        {
            if(clustbench_print_list_of_benchmarks(test_parameters.path_to_benchmark_code_dir))
            {
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }
        }
        MPI_Finalize();
        return 0;
    }

    if(error_flag == HELP_FLAG)
    {
        MPI_Finalize();
        return 0;
    }
    
    test_parameters.num_procs=comm_size;

    if (comm_size == 1)
    {
        if(comm_rank == 0)
        {
            printf("\n\nYou try to run this program with only one MPI process!\n\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    clustbench_benchmark_pointers_t pointers;
    if (clustbench_open_benchmark(test_parameters.path_to_benchmark_code_dir,
        test_parameters.benchmark_name,
        &pointers) != 0)
    {
        fprintf(stderr, "Cannot open the benchmark\n");
        return 1;
    }


    if(comm_rank == 0)
    {
        host_names = (char**)malloc(sizeof(char*)*comm_size);
        if(host_names == NULL)
        {
            fprintf(stderr,"Can't allocate memory %lu bytes for host_names\n",(unsigned long)(sizeof(char*)*comm_size));
            MPI_Abort(MPI_COMM_WORLD,1);
            return 1;
        }

        for ( i = 0; i < comm_size; i++ )
        {
            host_names[i] = (char*)malloc(CLUSTBENCH_HOSTNAME_LENGTH*sizeof(char));
            if(host_names[i]==NULL)
            {
                fprintf(stderr,"Can't allocate memory for name proc %d\n",i);
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }
        }
    } /* End if(rank==0) */

    /*
     * Going to get and write all processors' hostnames
     */

    if(clustbench_get_node_name(host_name))
    {
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }

    if (comm_rank == 0)
    {
        for ( i = 1; i < comm_size; i++ )
        {
            MPI_Recv(host_names[i], CLUSTBENCH_HOSTNAME_LENGTH, MPI_CHAR, i, 200, MPI_COMM_WORLD, &status);
        }
        strcpy(host_names[0],host_name);
    }
    else
    {
        MPI_Send(host_name, CLUSTBENCH_HOSTNAME_LENGTH, MPI_CHAR, 0, 200, MPI_COMM_WORLD);
    }

    if(comm_rank == 0)
    {
        /*
         *
         * Matrices initialization
         *
         */

        if(test_parameters.statistics_save & CLUSTBENCH_AVERAGE)
        {
            flag = easy_mtr_create(&mtr_av,comm_size,comm_size);
            if( flag==-1 )
            {
                fprintf(stderr,"Can not to create average matrix to story the test results\n");
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }

            if(create_netcdf_header(AVERAGE_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_av,&netcdf_var_av,pointers.define_netcdf_vars,pointers.put_netcdf_vars))
            {
                fprintf(stderr,"Can not to create file with name \"%s_average.nc\"\n",test_parameters.file_name_prefix);
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }
        }

        if(test_parameters.statistics_save & CLUSTBENCH_MEDIAN)
        {
            flag = easy_mtr_create(&mtr_me,comm_size,comm_size);
            if( flag==-1 )
            {
                fprintf(stderr,"Can not to create median matrix to story the test results\n");
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }

            if(create_netcdf_header(MEDIAN_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_me,&netcdf_var_me,pointers.define_netcdf_vars,pointers.put_netcdf_vars))
            {
                fprintf(stderr,"Can not to create file with name \"%s_median.nc\"\n",test_parameters.file_name_prefix);
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }
        }

        if(test_parameters.statistics_save & CLUSTBENCH_DEVIATION)
        {
            flag = easy_mtr_create(&mtr_di,comm_size,comm_size);
            if( flag==-1 )
            {
                fprintf(stderr,"Can not to create deviation matrix to story the test results\n");
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }

            if(create_netcdf_header(DEVIATION_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_di,&netcdf_var_di,pointers.define_netcdf_vars,pointers.put_netcdf_vars))
            {
                fprintf(stderr,"Can not to create file with name \"%s_deviation.nc\"\n",test_parameters.file_name_prefix);
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }
        }

        if(test_parameters.statistics_save & CLUSTBENCH_MIN)
        {
            flag = easy_mtr_create(&mtr_mi,comm_size,comm_size);
            if( flag==-1 )
            {
                fprintf(stderr,"Can not to create min values matrix to store the test results\n");
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }

            if(create_netcdf_header(MIN_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_mi,&netcdf_var_mi,pointers.define_netcdf_vars,pointers.put_netcdf_vars))
            {
                fprintf(stderr,"Can not to create file with name \"%s_min.nc\"\n",test_parameters.file_name_prefix);
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }
        }
        
        
        if(test_parameters.statistics_save & CLUSTBENCH_ALL)
        {
            flag = easy_mtr_create_3d(&mtr_all,comm_size,comm_size,test_parameters.num_repeats);
            if( flag==-1 )
            {
                fprintf(stderr,"Can not to create all values matrix to store  the test results\n");
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }

            if(create_netcdf_header_3d(ALL_DELAYS_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_all,&netcdf_var_all,pointers.define_netcdf_vars,pointers.put_netcdf_vars))
            {
                fprintf(stderr,"Can not to create file with name \"%s_all.nc\"\n",test_parameters.file_name_prefix);
                MPI_Abort(MPI_COMM_WORLD,1);
                return 1;
            }
        }

        if(clustbench_create_hosts_file(&test_parameters,host_names))    	
        {
    	    fprintf(stderr,"Can not to create file with name \"%s_hosts.txt\"\n",test_parameters.file_name_prefix);
        	MPI_Abort(MPI_COMM_WORLD,1);
        	return 1;
        }

        /*
         *
         * Printing initial message for user
         *
         */

        if(print_network_test_parameters(&test_parameters))
        {
            fprintf(stderr,"Can't print parameters\n");
            MPI_Abort(MPI_COMM_WORLD,1);
            return 1;
        }

    } /* End initialization  (only in MPI process with rank 0) */

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
        clustbench_time_result_t tmp_time;
        MPI_Get_address( &(tmp_time.average), &base);
        MPI_Get_address( &(tmp_time.median), &displace[1]);
        MPI_Get_address( &(tmp_time.deviation), &displace[2]);
        MPI_Get_address( &(tmp_time.min), &displace[3]);
    }
    displace[0]=0;
    displace[1]-=base;
    displace[2]-=base;
    displace[3]-=base;
    MPI_Type_create_struct(4,blocklength,displace,struct_types,&MPI_My_time_struct);
    MPI_Type_commit(&MPI_My_time_struct);


    times=(clustbench_time_result_t* )malloc(comm_size*sizeof(clustbench_time_result_t));
    if(times==NULL)
    {
    	fprintf(stderr, "Memory allocation error\n");
    	MPI_Abort(MPI_COMM_WORLD,1);
    	return 1;
    }
    
    //Новое
    real_times=(clustbench_time_t*)malloc(comm_size*test_parameters.num_repeats*sizeof(clustbench_time_t));
    if(real_times==NULL)
    {
        fprintf(stderr, "Memory allocation error\n");
    	MPI_Abort(MPI_COMM_WORLD,1);
    	return 1;
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
        
        pointers.test_function(times, real_times, tmp_mes_size, test_parameters.num_repeats, test_parameters.benchmark_parameters);

        MPI_Barrier(MPI_COMM_WORLD);

        if(comm_rank==0)
        {
            for(j=0; j<comm_size; j++)
            {
                if (test_parameters.statistics_save & CLUSTBENCH_AVERAGE)
                {
                    MATRIX_FILL_ELEMENT(mtr_av,0,j,times[j].average);
                }
                if (test_parameters.statistics_save & CLUSTBENCH_MEDIAN)
                {
                    MATRIX_FILL_ELEMENT(mtr_me,0,j,times[j].median);
                }
                if (test_parameters.statistics_save & CLUSTBENCH_DEVIATION)
                {
                    MATRIX_FILL_ELEMENT(mtr_di,0,j,times[j].deviation);
                }
                if (test_parameters.statistics_save & CLUSTBENCH_MIN)
                {
                    MATRIX_FILL_ELEMENT(mtr_mi,0,j,times[j].min);
                }
                //Новое
        
                if (test_parameters.statistics_save & CLUSTBENCH_ALL)
                {
                    for (int k = 0; k<test_parameters.num_repeats; k++){
                        MATRIX_FILL_ELEMENT_3D(mtr_all,0,j,k,real_times[j*test_parameters.num_repeats + k]);
                    }                    
                }
            }
            for(i=1; i<comm_size; i++)
            {
                MPI_Recv(times,comm_size,MPI_My_time_struct,i,100,MPI_COMM_WORLD,&status);
                MPI_Recv(real_times,comm_size*test_parameters.num_repeats,MPI_DOUBLE,i,101,MPI_COMM_WORLD,&status);
                for(j=0; j<comm_size; j++)
                {
                    if (test_parameters.statistics_save & CLUSTBENCH_AVERAGE)
                    {
                        MATRIX_FILL_ELEMENT(mtr_av,i,j,times[j].average);
                    }
                    if (test_parameters.statistics_save & CLUSTBENCH_MEDIAN)
                    {
                        MATRIX_FILL_ELEMENT(mtr_me,i,j,times[j].median);
                    }
                    if (test_parameters.statistics_save & CLUSTBENCH_DEVIATION)
                    {
                        MATRIX_FILL_ELEMENT(mtr_di,i,j,times[j].deviation);
                    }
                    if (test_parameters.statistics_save & CLUSTBENCH_MIN)
                    {
                        MATRIX_FILL_ELEMENT(mtr_mi,i,j,times[j].min);
                    }
                    //Новое
                    if (test_parameters.statistics_save & CLUSTBENCH_ALL)
                    {
                        for (int k = 0; k<test_parameters.num_repeats; k++){
                            MATRIX_FILL_ELEMENT_3D(mtr_all,i,j,k,real_times[j*test_parameters.num_repeats+k]);
                        }                    
                    }
                }
            }


            if (test_parameters.statistics_save & CLUSTBENCH_AVERAGE) 
            {
                if (netcdf_write_matrix(netcdf_file_av,netcdf_var_av,step_num,mtr_av.sizex,mtr_av.sizey,mtr_av.body))
                {
                    printf("Can't write average matrix to file.\n");
                    MPI_Abort(MPI_COMM_WORLD,1);
                    return 1;
                }
            }

            if (test_parameters.statistics_save & CLUSTBENCH_MEDIAN)
            {
                if (netcdf_write_matrix(netcdf_file_me,netcdf_var_me,step_num,mtr_me.sizex,mtr_me.sizey,mtr_me.body))
                {
                    printf("Can't write median matrix to file.\n");
                    MPI_Abort(MPI_COMM_WORLD,1);
                    return 1;
                }
            }

            if (test_parameters.statistics_save & CLUSTBENCH_DEVIATION)
            { 
                if (netcdf_write_matrix(netcdf_file_di,netcdf_var_di,step_num,mtr_di.sizex,mtr_di.sizey,mtr_di.body))
                {
                    printf("Can't write deviation matrix to file.\n");
                    MPI_Abort(MPI_COMM_WORLD,1);
                    return 1;
                }
            }

            if (test_parameters.statistics_save & CLUSTBENCH_MIN)
            {
                if (netcdf_write_matrix(netcdf_file_mi,netcdf_var_mi,step_num,mtr_mi.sizex,mtr_mi.sizey,mtr_mi.body))
                {
                    printf("Can't write matrix with minimal values to file.\n");
                    MPI_Abort(MPI_COMM_WORLD,1);
                    return 1;
                }
            }
            
            if (test_parameters.statistics_save & CLUSTBENCH_ALL)
            {
                if (netcdf_write_3d_matrix(netcdf_file_all,netcdf_var_all,step_num,mtr_all.sizex,mtr_all.sizey,mtr_all.sizez,mtr_all.body))
                {
                    printf("Can't write matrix with all values to file.\n");
                    MPI_Abort(MPI_COMM_WORLD,1);
                    return 1;
                }
            }
            


            printf("message length %d finished\r",tmp_mes_size);
            fflush(stdout);

        } /* end comm rank 0 */
        else
        {
            MPI_Send(times,comm_size,MPI_My_time_struct,0,100,MPI_COMM_WORLD);
            MPI_Send(real_times, comm_size*test_parameters.num_repeats, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
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
    if(comm_rank==0)
    {
        for (i = 0; i < comm_size; i++)
        {
            for (j = 0; j < comm_size; j++)
            {
                for (int k = 0; k < test_parameters.num_repeats; k++)
                {
                    printf("MAIN %d,%d,%d %e\n", i,j,k, MATRIX_GET_ELEMENT_3D(mtr_all,i,j,k));
                }
            }
        } 
    } 

    free(times);
    
    if (test_parameters.benchmark_parameters != NULL) 
    {
        pointers.free_parameters(&test_parameters);
    }
    
    clustbench_close_benchmark_lib(&pointers);

    if(comm_rank==0)
    {

        if (test_parameters.statistics_save & CLUSTBENCH_AVERAGE)
        {
            netcdf_close_file(netcdf_file_av);
            free(mtr_av.body);
        }
        if (test_parameters.statistics_save & CLUSTBENCH_MEDIAN)
        {
            netcdf_close_file(netcdf_file_me);
            free(mtr_me.body);
        }
        if (test_parameters.statistics_save & CLUSTBENCH_DEVIATION)
        {
            netcdf_close_file(netcdf_file_di);
            free(mtr_di.body);
        }
        if (test_parameters.statistics_save & CLUSTBENCH_MIN)
        {
            netcdf_close_file(netcdf_file_mi);
            free(mtr_mi.body);
        }
        if (test_parameters.statistics_save & CLUSTBENCH_ALL)
        {
            netcdf_close_file(netcdf_file_all);
            free(mtr_all.body);
        }

        for(i=0; i<comm_size; i++)
        {
            free(host_names[i]);
        }
        free(host_names);

        printf("\nTest is done\n");
    }

    MPI_Finalize();
    return 0;
} /* main finished */

