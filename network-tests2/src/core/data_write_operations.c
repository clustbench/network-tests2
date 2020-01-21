#include <netcdf.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <fcntl.h>

#include "types.h"
#include "data_write_operations.h"
#include "string_id_converters.h"
#define TDIMS 2 




int create_netcdf_header(
		const int file_data_type, 
		const struct network_test_parameters_struct *test_parameters, 
		int *file_id, 
		int *data_id/*, 
        MPI_Comm comm, 
        MPI_Info info*/)
{

	int netcdf_file_id;
	int x_dim_id, y_dim_id, num_matrices_dim_id, strings_dim_id;
	int num_procs_var_id, test_type_var_id, file_data_type_var_id, begin_message_length_var_id, end_message_length_var_id;

	int step_length_var_id, num_repeats_var_id;
    int noise_message_length_var_id, num_noise_messages_var_id, num_noise_procs_var_id;
    int data_var_id;
	int dims[3];
    int d;
    
    int retval;
	char str[25];
	char *file_name = NULL;
	size_t tx_start[TDIMS];
     	size_t tx_count[TDIMS];

    file_name = (char*)malloc(strlen(test_parameters->file_name_prefix)+strlen("_daviation.nc")+1);
    
	if(file_name == NULL)
	{
		return MEM_ERROR;
	}

	sprintf(file_name,"%s_%s.nc",test_parameters->file_name_prefix,file_data_type_to_sring(file_data_type));


    if ((retval = nc_create(file_name, NC_WRITE | NC_NOCLOBBER | NC_64BIT_OFFSET | NC_SHARE, &netcdf_file_id))!= NC_NOERR)
        return CREATE_FILE_ERROR;
    //if ((retval = nc_create_par(file_name,NC_WRITE | NC_NOCLOBBER | NC_64BIT_OFFSET | NC_SHARE,comm,info, &netcdf_file_id))!= NC_NOERR)
        //return CREATE_FILE_ERROR;
	
	free(file_name);
	file_name = NULL;

//Set dimensions

	if ((retval = nc_def_dim(netcdf_file_id, "x",1 /*test_parameters->num_procs*/,&x_dim_id))!=NC_NOERR)
		return NETCDF_ERROR;
	
	if ((retval = nc_def_dim(netcdf_file_id, "y", test_parameters->num_procs,&y_dim_id))!=NC_NOERR)
                return NETCDF_ERROR;

	if ((retval = nc_def_dim(netcdf_file_id, "n", NC_UNLIMITED,&num_matrices_dim_id))!=NC_NOERR)
                return NETCDF_ERROR;

	if ((retval = nc_def_dim(netcdf_file_id, "strings", 101L,&strings_dim_id))!=NC_NOERR)
                return NETCDF_ERROR;
	

//Set variables
    dims[0] = num_matrices_dim_id;
	dims[1] = x_dim_id;
	dims[2] = y_dim_id;

	if ((retval = nc_def_var(netcdf_file_id, "proc_num", NC_INT,0,0,&num_procs_var_id))!=NC_NOERR)
		return NETCDF_ERROR;
    
	if ((retval = nc_def_var(netcdf_file_id, "test_type", NC_CHAR, 1, &strings_dim_id, &test_type_var_id))!= NC_NOERR)
		return NETCDF_ERROR;
    

	if ((retval = nc_def_var(netcdf_file_id, "data_type", NC_CHAR, 1, &strings_dim_id, &file_data_type_var_id))!= NC_NOERR)
                return NETCDF_ERROR;
    

	if ((retval = nc_def_var(netcdf_file_id, "begin_mes_length", NC_INT, 0, 0, &begin_message_length_var_id))!= NC_NOERR)
                return NETCDF_ERROR;
 

	if ((retval = nc_def_var(netcdf_file_id, "end_mes_length", NC_INT, 0, 0, &end_message_length_var_id))!= NC_NOERR)
                return NETCDF_ERROR;
    

	if ((retval = nc_def_var(netcdf_file_id, "step_length", NC_INT, 0, 0, &step_length_var_id))!= NC_NOERR)
                return NETCDF_ERROR;
    
	if ((retval = nc_def_var(netcdf_file_id, "noise_mes_length", NC_INT, 0, 0, &noise_message_length_var_id))!= NC_NOERR)
                return NETCDF_ERROR;
    
   

	if ((retval = nc_def_var(netcdf_file_id, "num_noise_mes", NC_INT, 0, 0, &num_noise_messages_var_id))!= NC_NOERR)
                return NETCDF_ERROR;


	if ((retval = nc_def_var(netcdf_file_id, "num_noise_proc", NC_INT, 0, 0, &num_noise_procs_var_id))!= NC_NOERR)
                return NETCDF_ERROR;
 

	if ((retval = nc_def_var(netcdf_file_id, "num_repeates", NC_INT, 0, 0, &num_repeats_var_id))!= NC_NOERR)
                return NETCDF_ERROR;
  
    
    
	dims[0] = num_matrices_dim_id;
	dims[1] = x_dim_id;
	dims[2] = y_dim_id;

	if ((retval = nc_def_var(netcdf_file_id, "data", NC_DOUBLE, 3, dims, &data_var_id))!=NC_NOERR)
		return NETCDF_ERROR;

//Set var att
    if (str == NULL)
    {
        printf("NULL\n");
    }
    else
    {
        get_test_type_name(test_parameters->test_type, str);
    }


	//strcpy(data[0], str);
	//printf("%s\n", data[0]);
	/*int i;
	for(i=0;i<strlen(str);i++)
	{
		data[i] = str[i];
	}*/
	
    //str[0] = (NC_CHAR)"a";
    //str = (char*)realloc(str,strlen(str)+sizeof(char));
    //str[strlen(str)] = '\0';
    
/*	if ((retval = nc_put_att_text(netcdf_file_id,test_type_var_id, "test_type",strlen(str),str))!=NC_NOERR)
		return NETCDF_ERROR;
	
	if ((retval = nc_put_att_text(netcdf_file_id,file_data_type_var_id, "data_type", strlen(file_data_type_to_sring(file_data_type)),file_data_type_to_sring(file_data_type)))!=NC_NOERR)
		return NETCDF_ERROR;

	if((retval = nc_put_att_int(netcdf_file_id,num_procs_var_id,"num_procs",NC_INT,1,&test_parameters->num_procs))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_att_int(netcdf_file_id,begin_message_length_var_id,"begin_mes_length", NC_INT,1,&test_parameters->begin_message_length))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_att_int(netcdf_file_id,end_message_length_var_id,"end_mes_length",NC_INT,1,&test_parameters->end_message_length))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_att_int(netcdf_file_id,step_length_var_id,"step_length",NC_INT,1,&test_parameters->step_length))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_att_int(netcdf_file_id,num_repeats_var_id,"num_repeats",NC_INT,1,&test_parameters->num_repeats))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_att_int(netcdf_file_id,noise_message_length_var_id,"noise_mes_length",NC_INT,1,&test_parameters->noise_message_length))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_att_int(netcdf_file_id,num_noise_procs_var_id,"num_noise_proc",NC_INT,1,&test_parameters->num_noise_procs))!=NC_NOERR)
                return NETCDF_ERROR;
	
	if((retval = nc_put_att_int(netcdf_file_id,num_noise_messages_var_id,"num_noise_mes",NC_INT,1,&test_parameters->num_noise_messages))!=NC_NOERR)
                return NETCDF_ERROR;
 */   
//Border
    
	if((retval = nc_enddef(netcdf_file_id))!=NC_NOERR)
                return NETCDF_ERROR;

	tx_start[0] = 0;      /* write only */
     	tx_start[1] = 0;      /* one line*/
     	tx_count[0] = strlen(str);      /* write that  */
     	tx_count[1] = strlen(str) + 1;  /* much symbols */
	retval = nc_put_vara_text(netcdf_file_id, test_type_var_id, tx_start, tx_count, str);
	if (retval!=NC_NOERR)
		return NETCDF_ERROR;
	
	tx_count[0] = strlen(file_data_type_to_sring(file_data_type));      
     	tx_count[1] = strlen(file_data_type_to_sring(file_data_type)) + 1;   
	retval = nc_put_vara_text(netcdf_file_id, file_data_type_var_id, tx_start, tx_count, file_data_type_to_sring(file_data_type));
	if (retval!=NC_NOERR)
		return NETCDF_ERROR;
   /* if((retval = nc_put_var_text(netcdf_file_id,file_data_type_var_id,(char *)file_data_type_to_sring(file_data_type)))!=NC_NOERR)
        return NETCDF_ERROR;*/

	if((retval = nc_put_var_int(netcdf_file_id,num_procs_var_id,&test_parameters->num_procs))!=NC_NOERR)
                return NETCDF_ERROR;

	if((retval = nc_put_var_int(netcdf_file_id,begin_message_length_var_id,&test_parameters->begin_message_length))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_var_int(netcdf_file_id,end_message_length_var_id,&test_parameters->end_message_length))!=NC_NOERR)
                return NETCDF_ERROR;

	if((retval = nc_put_var_int(netcdf_file_id,step_length_var_id,&test_parameters->step_length))!=NC_NOERR)
                return NETCDF_ERROR;

	if((retval = nc_put_var_int(netcdf_file_id,num_repeats_var_id,&test_parameters->num_repeats))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_var_int(netcdf_file_id,noise_message_length_var_id,&test_parameters->noise_message_length))!=NC_NOERR)
                return NETCDF_ERROR;

    if((retval = nc_put_var_int(netcdf_file_id,num_noise_procs_var_id,&test_parameters->num_noise_procs))!=NC_NOERR)
                return NETCDF_ERROR;

	if((retval = nc_put_var_int(netcdf_file_id,num_noise_messages_var_id,&test_parameters->num_noise_messages))!=NC_NOERR)
                return NETCDF_ERROR;
    
    
    //if((retval = nc_redef(netcdf_file_id))!=NC_NOERR)
    //            return NETCDF_ERROR;
    
	nc_sync(netcdf_file_id);

	*file_id = netcdf_file_id;
	*data_id = data_var_id;
    
    if (retval = nc_close(netcdf_file_id))
            return NETCDF_ERROR;
	
    return 0;
}

int open_netcdf_file(const int file_data_type, const struct network_test_parameters_struct *test_parameters, int *file_id, int *data_id)
{
    char *file_name = NULL;
    int retval, netcdf_file_id, data_var_id;
    int old_fill_mode;
    
    file_name = (char*)malloc(strlen(test_parameters->file_name_prefix)+strlen("_deviation.nc")+1);
    
    if(file_name == NULL)
	{
		return MEM_ERROR;
	}
	
    sprintf(file_name,"%s_%s.nc",test_parameters->file_name_prefix,file_data_type_to_sring(file_data_type));
    
    if ((retval = nc_open(file_name, NC_WRITE |  NC_SHARE /*|NC_NOCLOBBER |NC_64BIT_OFFSET */, &netcdf_file_id))!= NC_NOERR)
        return NETCDF_ERROR;
    
    free(file_name);
	//file_name = NULL;
    //if((retval = nc_redef(netcdf_file_id))!=NC_NOERR)
    //        return NETCDF_ERROR;
    if ((retval = nc_inq_varid(netcdf_file_id, "data", &data_var_id))!= NC_NOERR)
        return NETCDF_ERROR;
    if ((retval = nc_set_fill(netcdf_file_id, NC_NOFILL, &old_fill_mode))!=NC_NOERR)
            return NETCDF_ERROR;
    //if((retval = nc_enddef(netcdf_file_id))!=NC_NOERR)
     //           return NETCDF_ERROR;
    
    //if((retval = nc_redef(netcdf_file_id))!=NC_NOERR)
     //           return NETCDF_ERROR;
    //if((retval = nc_enddef(netcdf_file_id))!=NC_NOERR)
     //           return NETCDF_ERROR;
    
    
    *file_id = netcdf_file_id;
    *data_id = data_var_id;
    
    return 0;
}

int netcdf_write_matrix
(
         int netcdf_file_id,
         int netcdf_var_id,
        const int matrix_number_in_file,
        const int size_x,
        const int size_y,
        const double *data,
        const int comm_rank,
        const int file_data_type,
        const struct network_test_parameters_struct* test_parameters
)
{
        size_t start[3]={matrix_number_in_file,0,0};
        size_t count[3]={1,size_x,size_y};
        int retval, old_fill_mode;
        
        
        

        //if (retval = open_netcdf_file(file_data_type,test_parameters,&netcdf_file_id,&netcdf_var_id))
        //    return NETCDF_ERROR;
            
        //nc_sync(netcdf_file_id);
        
       // if((retval = nc_redef(netcdf_file_id))!=NC_NOERR)
        //        return NETCDF_ERROR;
        
       // if ((retval = nc_inq_varid(netcdf_file_id, "data", &netcdf_var_id))!= NC_NOERR)
        //    return NETCDF_ERROR;
        
        //if((retval = nc_enddef(netcdf_file_id))!=NC_NOERR)
        //        return NETCDF_ERROR;
        
        //if ((retval = nc_set_fill(netcdf_file_id, NC_NOFILL, &old_fill_mode))!=NC_NOERR)
        //    return NETCDF_ERROR;
        
        nc_sync(netcdf_file_id);
        if(nc_put_vara_double(netcdf_file_id,netcdf_var_id,start,count,data)!=NC_NOERR)
        {
                return -35;
        }
        
       // nc_sync(netcdf_file_id);
        
        /*if((retval = nc_redef(netcdf_file_id))!=NC_NOERR)
                return NETCDF_ERROR;
        
        if ((retval = nc_inq_varid(netcdf_file_id, "data", &netcdf_var_id))!= NC_NOERR)
            return NETCDF_ERROR;
        //nc_sync(netcdf_file_id);
        if((retval = nc_enddef(netcdf_file_id))!=NC_NOERR)
                return NETCDF_ERROR;*/
        
        //if((retval = nc_redef(netcdf_file_id))!=NC_NOERR)
         //       return NETCDF_ERROR;
        //if((retval = nc_enddef(netcdf_file_id))!=NC_NOERR)
         //       return NETCDF_ERROR;
         
        //nc_sync(netcdf_file_id);
        
        
        return 0;
}

int netcdf_close_file(const int netcdf_file_id)
{
    int i,j;
    int retval;
    //nc_sync(netcdf_file_id);
    if (retval = nc_close(netcdf_file_id))
        return NETCDF_ERROR;
    
    return 0;
}

