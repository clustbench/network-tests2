/*  This file is a part of the PARUS project.
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
#include "libconfig.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include "dlfcn.h"

#ifdef _GNU_SOURCE
#include <getopt.h>
#else
#include <unistd.h>
#endif

#include "my_time.h"
#include "my_malloc.h"
#include "parus_config.h"
#include "comm_proc.h"
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
    char library[100] = "\0";
    char library1[100] = "\0";
    char tests[10] = "tests/";
    char libso[5] = ".so";
    char liblib[5] ="/lib";
    char *sbuf;
    //char conf[15] = "/proc_config.cfg";
    char test[100];
	void *handle;
    void (*func_print_name)(Test_time_result_type*, int,int);
	void (*func1)(Test_time_result_type*, int,int,int,int,int, char**, COMM_PROC*);
    FILE *f;
    char d[30] = "\0";
    char txt[5] = ".txt";
    char papka[100]="\0";
    
    const char *noise_proc;
    int c, m;
    int len;
    char b;
    config_t cfg;
    
    char **proc_names= NULL;
    char **proc_names1 = NULL;
    
    char *hosts_name;
    char *host_name1;
    
    int kol;
    int num_noise_hosts;
    
    int flag_proc= 0;
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

    int step_num_av=0;
    int step_num_min=0;
    int step_num_de=0;
    int step_num_me=0;
    
    int comm_rank_av;
    int comm_rank_min; 
    int comm_rank_med; 
    int comm_rank_dev;
    
    int comm_rank_file;
    int comm_size_file;
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
    
    int len_dest_file;
    char *dest_file = NULL;

    char** host_names=NULL;
    char host_name[256];

    int comm_rank_prob;
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

    MPI_Info info= MPI_INFO_NULL;
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
    
    
    COMM_PROC comm_proc[comm_size];
    
    if(comm_rank == 0)
    {
        if ( comm_size == 1 )
        {
            error_flag = 1;
            printf( "\n\nYou tries to run this programm for one MPI thread!\n\n" );
        }
        
        printf("\n"); 
        if (parse_network_test_arguments(argc,argv,&test_parameters))
        {
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
        
        //printf("all right 1 \n");                
        
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

    MPI_Barrier(MPI_COMM_WORLD);
    gethostname( host_name, 255 );
    
    if ( comm_rank == 0 )
    {
        for ( i = 1; i < comm_size; i++ )
        {
            MPI_Recv( host_names[i], 256, MPI_CHAR, i, 200, MPI_COMM_WORLD, &status );
        }
        
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
    //if( comm_rank == 0)
    //{ 
        /*
         *
         * Matrices initialization
         *
         */
    
    
    flag = easy_mtr_create(&mtr_av, 1, comm_size);
    if( flag==-1 )
    {
        printf("Can not to create average matrix to story the test results\n");
        MPI_Abort(MPI_COMM_WORLD,-1);
        return -1;
    }
    flag = easy_mtr_create(&mtr_me,1,comm_size);
        if( flag==-1 )
        {
            printf("Can not to create median matrix to story the test results\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    flag = easy_mtr_create(&mtr_di,1,comm_size);
        if( flag==-1 )
        {
            printf("Can not to create deviation matrix to story the test results\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    flag = easy_mtr_create(&mtr_mi,1,comm_size);
        if( flag==-1 )
        {
            printf("Can not to create min values matrix to story  the test results\n");
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
        
    if( comm_rank == 0)
    { 
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
        if ((test_parameters.test_type == NOISE_TEST_TYPE) || (test_parameters.test_type == NOISE_BLOCKING_TEST_TYPE))
        {
            printf("\tnoise message length\t\t%d\n",test_parameters.noise_message_length);
            printf("\tnumber of noise messages\t%d\n",test_parameters.num_noise_messages);
            printf("\tnumber of noise processes\t%d\n",test_parameters.num_noise_procs);
        }
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
    
    
    //Delivering file_name to all processes
    if (comm_rank==0)
    {
        len_dest_file = strlen(test_parameters.file_name_prefix);
        dest_file = (char*)malloc(strlen(test_parameters.file_name_prefix)*sizeof(char));
        dest_file[0] = '\0';
        strcat(dest_file, test_parameters.file_name_prefix);
    }

    MPI_Bcast(&len_dest_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (comm_rank!=0)
    {
        dest_file = (char*)malloc(len_dest_file*sizeof(char));
        dest_file[0] = '\0';
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(dest_file, len_dest_file, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (comm_rank!=0)
    {
        dest_file = (char*)realloc(dest_file, len_dest_file+3*sizeof(char));
        dest_file[len_dest_file] = '\0';
    }
    MPI_Barrier(MPI_COMM_WORLD);

    test_parameters.file_name_prefix = dest_file;
    
    
    //Open files with max 4 procceses
    comm_rank_file = 0;
    if (comm_size > 4)
        comm_size_file = 4;
    else
        comm_size_file = comm_size;
    MPI_Barrier(MPI_COMM_WORLD);
    
    //if (comm_rank == 1 || comm_rank==0)
    //{
    if (open_netcdf_file(AVERAGE_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_av,&netcdf_var_av))
        {
            printf("Can not to open file with name \"%s_average.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    if (open_netcdf_file(MEDIAN_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_me,&netcdf_var_me))
        {
            printf("Can not to open file with name \"%s_median.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    if (open_netcdf_file(DEVIATION_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_di,&netcdf_var_di))
        {
            printf("Can not to open file with name \"%s_deviation.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    if (open_netcdf_file(MIN_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_mi,&netcdf_var_mi))
        {
            printf("Can not to open file with name \"%s_min.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
   // }
    //Open files for parallel writing
    /*if (comm_rank != 0)
    {*/
        //printf("%d\n", test_parameters.test_type);
   /* if (comm_rank_file == comm_rank)
    {
        if (open_netcdf_file(AVERAGE_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_av,&netcdf_var_av))
        {
            printf("Can not to open file with name \"%s_average.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    }
    comm_rank_file++;
    if (comm_rank_file == comm_size_file)
        comm_rank_file = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank_file == comm_rank)
    {
        
        if (open_netcdf_file(MEDIAN_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_me,&netcdf_var_me))
        {
            printf("Can not to open file with name \"%s_median.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    }
    comm_rank_file++;
    if (comm_rank_file == comm_size_file)
        comm_rank_file = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank_file == comm_rank)
    {
        if (open_netcdf_file(DEVIATION_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_di,&netcdf_var_di))
        {
            printf("Can not to open file with name \"%s_deviation.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    }
    comm_rank_file++;
    if (comm_rank_file == comm_size_file)
        comm_rank_file = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank_file == comm_rank)
    {
        if (open_netcdf_file(MIN_NETWORK_TEST_DATATYPE,&test_parameters,&netcdf_file_mi,&netcdf_var_mi))
        {
            printf("Can not to open file with name \"%s_min.nc\"\n",test_parameters.file_name_prefix);
            MPI_Abort(MPI_COMM_WORLD,-1);
            return -1;
        }
    }*/
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (test_parameters.test_type == NOISE_TEST_TYPE || test_parameters.test_type ==NOISE_BLOCKING_TEST_TYPE)
    {
        gethostname( host_name, 255 );
    
    //MPI_Barrier(MPI_COMM_WORLD);
        for (i=0;i<comm_size;i++)
        {
            MPI_Send( host_name, 256, MPI_CHAR, i, 200, MPI_COMM_WORLD );
        }
    
        for (i=0;i<comm_size;i++)
        {
            if (i != comm_rank)
            {
                host_name[0] = '\0';
                MPI_Recv( host_name, 256, MPI_CHAR, i, 200, MPI_COMM_WORLD, &status );
                comm_proc[i].host_name[0] = '\0';
                strcat(comm_proc[i].host_name,host_name);
                comm_proc[i].num_proc = i;
            }
            else
            {
                host_name[0] = '\0';
                gethostname( host_name, 255 );
                comm_proc[i].host_name[0] = '\0';
                strcat(comm_proc[i].host_name,host_name);
                comm_proc[i].num_proc = i;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    

    if (test_parameters.test_type == NOISE_BLOCKING_TEST_TYPE || test_parameters.test_type == NOISE_TEST_TYPE)
        {
            config_init(&cfg);
            if (!config_read_file(&cfg, "proc_config.cfg"))
            {
                return -1;
            }
        
            if (config_lookup_string(&cfg, "procs", &noise_proc))
            {
                m=0;
                kol=0;
                host_name1 = (char*)malloc(1*sizeof(char));
                if (host_name1 == NULL)
                {
                    return -1;
                }
                hosts_name = host_name1;
                for (c=0;c<strlen(noise_proc)+1; c++)
                {
                    if (noise_proc[c] == '\0')
                    {
                        hosts_name[kol] = '\0';
                        proc_names1  = (char**)realloc(proc_names, (m+1)*sizeof(char));
                        if (proc_names1 == NULL)
                        {
                            printf("realloc error\n");
                            return 1;
                        }
                        proc_names = proc_names1;
                        proc_names[m] = hosts_name;
                        m++;
                        num_noise_hosts = m;
                        //host_name1 = (char*)malloc(1*sizeof(char));
                        //hosts_name = host_name1;
                        break;
                    }
                    if (noise_proc[c] == '\n')
                    {
                        hosts_name[kol] = '\0';
                        proc_names1  = (char**)realloc(proc_names, (m+1)*sizeof(char*));
                        if (proc_names1 == NULL)
                        {
                            printf("realloc error\n");
                            return 1;
                        }
                        proc_names = proc_names1;
                        proc_names[m] = hosts_name;
                        m++;
                        host_name1 = (char*)malloc(1*sizeof(char));
                        hosts_name = host_name1;
                        kol=0;
                        continue;
                    }
                    hosts_name[kol] = noise_proc[c];
                    host_name1 = (char*)realloc(hosts_name, (kol+1)*sizeof(char*));
                    if (host_name1 == NULL)
                    {
                        return -1;
                    }
                    hosts_name = host_name1;
                    kol++;
                }
            }
                //printf("%s\n", noise_procs);
            
            config_destroy(&cfg);
            
            
        }
        
    
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
    int k=0;
    get_test_type_name(test_parameters.test_type,test);
    strcat(library, tests);
    strcat(library, test);
    strcat(library, liblib);
    strcat(library, test);
    strcat(library, libso);
    handle = dlopen(library, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error: %s\n", dlerror());
        		return EXIT_FAILURE;
    		}
    if (strcmp(library, "tests/noise/libnoise.so")==0 || strcmp(library, "tests/noise/libnoise_blocking.so") ==0) {
        k=1;
        
        *(void**)(&func1) = dlsym(handle, test);
        if (!func1) {
        		
        		fprintf(stderr, "Error: %s\n", dlerror());
        		dlclose(handle);
        		return EXIT_FAILURE;
    		}
    }
    else {
        k=0;
        *(void**)(&func_print_name) = dlsym(handle, test);
        if (!func_print_name) {
        		
        		fprintf(stderr, "Error: %s\n", dlerror());
        		dlclose(handle);
        		return EXIT_FAILURE;
    		}
    }
    
    for
	    (
	     tmp_mes_size=test_parameters.begin_message_length;
	     tmp_mes_size<((test_parameters.end_message_length) /*- test_parameters.step_length)/*comm_size*/);
	     step_num+=comm_size,tmp_mes_size+=test_parameters.step_length
	     )
    {
        
        if (k==1)
        {
            //MPI_Barrier(MPI_COMM_WORLD);
            func1(
                times,
                tmp_mes_size,
                test_parameters.num_repeats,
                test_parameters.num_noise_messages,
                test_parameters.noise_message_length,
		       	test_parameters.num_noise_procs,
                proc_names,
                comm_proc);
        }
        else
        {
            func_print_name(times,tmp_mes_size,test_parameters.num_repeats);
        }
        
        //MPI_Barrier(MPI_COMM_WORLD);
        for(j=0; j<comm_size;j++)
        {
            MATRIX_FILL_ELEMENT(mtr_av,0,j,times[j].average);
            MATRIX_FILL_ELEMENT(mtr_me,0,j,times[j].median);
            MATRIX_FILL_ELEMENT(mtr_di,0,j,times[j].deviation);
            MATRIX_FILL_ELEMENT(mtr_mi,0,j,times[j].min);
        }
        //printf("comm_rank %d  %d  %d\n",comm_rank,mtr_av.sizex,mtr_av.sizey); 
        //if (comm_rank == 1 || comm_rank == 0)
        //{
        MPI_Barrier(MPI_COMM_WORLD);
        if(netcdf_write_matrix(netcdf_file_av,netcdf_var_av,step_num+comm_rank,mtr_av.sizex,mtr_av.sizey,mtr_av.body, comm_rank, AVERAGE_NETWORK_TEST_DATATYPE, &test_parameters))  //1
        {
                printf("Can't write average matrix to file.\n");
                MPI_Abort(MPI_COMM_WORLD,-1);
                return 1;
        }

        if(netcdf_write_matrix(netcdf_file_me,netcdf_var_me,step_num+comm_rank,mtr_me.sizex,mtr_me.sizey,mtr_me.body,comm_rank, MEDIAN_NETWORK_TEST_DATATYPE, &test_parameters))  //2
        {
                printf("Can't write median matrix to file.\n");
                MPI_Abort(MPI_COMM_WORLD,-1);
                return 1;
        }

        if(netcdf_write_matrix(netcdf_file_di,netcdf_var_di,step_num+comm_rank,mtr_di.sizex,mtr_di.sizey,mtr_di.body,comm_rank, DEVIATION_NETWORK_TEST_DATATYPE, &test_parameters))  //3
        {
                printf("Can't write deviation matrix to file.\n");
                MPI_Abort(MPI_COMM_WORLD,-1);
                return 1;
        }

        if(netcdf_write_matrix(netcdf_file_mi,netcdf_var_mi,step_num+comm_rank,mtr_mi.sizex,mtr_mi.sizey,mtr_mi.body,comm_rank, MIN_NETWORK_TEST_DATATYPE, &test_parameters))  //4
        {
                printf("Can't write  matrix with minimal values to file.\n");
                MPI_Abort(MPI_COMM_WORLD,-1);
                return 1;
        }
        
       
        if (comm_rank==0)
        {
            printf("message length %d finished\r",tmp_mes_size);
            fflush(stdout);

        }
        //MPI_Barrier(MPI_COMM_WORLD);
        


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
    MPI_Barrier(MPI_COMM_WORLD);

    /*free(times);
    if (test_parameters.test_type == NOISE_BLOCKING_TEST_TYPE || test_parameters.test_type == NOISE_TEST_TYPE)
    {
        for (i=0;i<num_noise_hosts;i++)
        {
            free(proc_names[i]);
        }
        free(proc_names);
    }*/
    

    netcdf_close_file(netcdf_file_av);
    netcdf_close_file(netcdf_file_me);
    netcdf_close_file(netcdf_file_di);
    netcdf_close_file(netcdf_file_mi);

    free(mtr_av.body);
    free(mtr_me.body);
    free(mtr_di.body);
    free(mtr_mi.body);
    
	if(comm_rank==0)
    {

        
        /*free(mtr_av.body);
        free(mtr_me.body);
        free(mtr_di.body);
        free(mtr_mi.body);
        */
        for(i=0; i<comm_size; i++)
        {
            free(host_names[i]);
        }
        free(host_names);

        /*free(mtr_av.body);
        free(mtr_me.body);
        free(mtr_di.body);
        free(mtr_mi.body);*/

        printf("\nTest is done\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
} /* main finished */

