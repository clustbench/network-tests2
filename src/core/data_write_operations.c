#include <netcdf.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "types.h"
#include "data_write_operations.h"
#include "string_id_converters.h"




int create_netcdf_header
(
	const int file_data_type,
	const struct network_test_parameters_struct *test_parameters,
	int *file_id,
	int *data_id
)
{
	int netcdf_file_id;

	int x_dim_id, y_dim_id, num_matrices_dim_id, strings_dim_id;

	int num_procs_var_id, test_type_var_id, file_data_type_var_id, begin_message_length_var_id, end_message_length_var_id;
	int step_length_var_id, num_repeats_var_id;
	int noise_mesage_length_var_id, num_noise_messages_var_id, num_noise_procs_var_id;	
	int data_var_id;

	int dims[3];

	

	char *file_name=NULL;

	file_name=(char *)malloc(strlen(test_parameters->file_name_prefix)+strlen("_deviation.nc")+1);
	if(file_name==NULL)
	{
		return MEM_ERROR;
	}

	sprintf(file_name,"%s_%s.nc",test_parameters->file_name_prefix,file_data_type_to_sring(file_data_type));

	if(nc_create(file_name,NC_NOCLOBBER|NC_SHARE|NC_64BIT_OFFSET,&netcdf_file_id)!=NC_NOERR)
	{
		return CREATE_FILE_ERROR;
	}
	free(file_name);
	file_name=NULL;

	if(nc_def_dim(netcdf_file_id,"x",test_parameters->num_procs,&x_dim_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_def_dim(netcdf_file_id,"y",test_parameters->num_procs,&y_dim_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_def_dim(netcdf_file_id,"n",NC_UNLIMITED,&num_matrices_dim_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_def_dim(netcdf_file_id,"strings",101,&strings_dim_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }

	if(nc_def_var(netcdf_file_id,"proc_num",NC_INT,0,0,&num_procs_var_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}
	
	/*
         * For future
         */
	/*
	if(nc_def_var(netcdf_file_id,"test_type",NC_CHAR,1,&strings_dim_id,&test_type_var_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }
        */

	if(nc_def_var(netcdf_file_id,"test_type",NC_INT,0,0,&test_type_var_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }

	/*
         * For future
         */
	/*
	if(nc_def_var(netcdf_file_id,"data_type",NC_CHAR,1,&strings_dim_id,&file_data_type_var_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }
        */

	if(nc_def_var(netcdf_file_id,"data_type",NC_INT,0,0,&file_data_type_var_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }

	if(nc_def_var(netcdf_file_id,"begin_mes_length",NC_INT,0,0,&begin_message_length_var_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }

	if(nc_def_var(netcdf_file_id,"end_mes_length",NC_INT,0,0,&end_message_length_var_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }

	if(nc_def_var(netcdf_file_id,"step_length",NC_INT,0,0,&step_length_var_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }

	if(nc_def_var(netcdf_file_id,"noise_mes_length",NC_INT,0,0,&noise_mesage_length_var_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
        }

	if(nc_def_var(netcdf_file_id,"num_noise_mes",NC_INT,0,0,&num_noise_messages_var_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
        }

	if(nc_def_var(netcdf_file_id,"num_noise_proc",NC_INT,0,0,&num_noise_procs_var_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
        }

	if(nc_def_var(netcdf_file_id,"num_repeates",NC_INT,0,0,&num_repeats_var_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
        }

	dims[0]=num_matrices_dim_id;
	dims[1]=x_dim_id;
	dims[2]=y_dim_id;
	if(nc_def_var(netcdf_file_id,"data",NC_DOUBLE,3,dims,&data_var_id)!=NC_NOERR)
        {
                return NETCDF_ERROR;
        }
	
	if(nc_enddef(netcdf_file_id)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,num_procs_var_id,&test_parameters->num_procs)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,test_type_var_id,&test_parameters->test_type)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,file_data_type_var_id,&file_data_type)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,begin_message_length_var_id,&test_parameters->begin_message_length)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,end_message_length_var_id,&test_parameters->end_message_length)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,step_length_var_id,&test_parameters->step_length)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,num_repeats_var_id,&test_parameters->num_repeats)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,noise_mesage_length_var_id,&test_parameters->noise_message_length)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,num_noise_messages_var_id,&test_parameters->num_noise_messages)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	if(nc_put_var_int(netcdf_file_id,num_noise_procs_var_id,&test_parameters->num_noise_procs)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}

	nc_sync(netcdf_file_id);
	
	*file_id=netcdf_file_id;
	*data_id=data_var_id;
	return 0;	
}

int netcdf_write_matrix
(
	const int netcdf_file_id,
	const int netcdf_var_id,
	const int matrix_number_in_file,
	const int size_x,
	const int size_y,
	const double *data
)
{
	size_t start[3]={matrix_number_in_file,0,0};
	size_t count[3]={1,size_x,size_y};
	if(nc_put_vara_double(netcdf_file_id,netcdf_var_id,start,count,data)!=NC_NOERR)
	{
		return NETCDF_ERROR;
	}
	nc_sync(netcdf_file_id);

	return 0;
}

int netcdf_close_file(const int netcdf_file_id)
{
	nc_close(netcdf_file_id);
	return 0;
}
