#ifndef   __DATA_WRITE_OPERATIONS_H__
#define   __DATA_WRITE_OPERATIONS_H__

#include "types.h"

#define MEM_ERROR                -2
#define CREATE_FILE_ERROR        -3
#define NETCDF_ERROR             -4


/**
 * This function writes header for NetCDF file that contains 
 * results recieved after the test execution on the 
 * Multiprocessor.
 *
 * If all was OK you will see filled parameters file_id with NetCDF id and data_id with NetCDF data variable id.
 * return value is 0 on success and negative value on error. 
 *
 */

#ifdef __cplusplus
extern "C"
{
#endif

extern int create_netcdf_header
(
	const int file_data_type, 
	const struct network_test_parameters_struct* test_parameters,
	int *file_id,
	int *data_id
);

extern int netcdf_write_matrix
(
        const int netcdf_file_id,
        const int netcdf_var_id,
        const int matrix_number_in_file,
        const int size_x,
        const int size_y,
        const double *data
);

extern int netcdf_close_file(const int netcdf_file_id);

#ifdef __cplusplus
}
#endif

#endif  /* __DATA_WRITE_OPERATIONS_H__ */
