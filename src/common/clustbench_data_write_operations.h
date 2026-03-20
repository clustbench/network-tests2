#ifndef   __DATA_WRITE_OPERATIONS_H__
#define   __DATA_WRITE_OPERATIONS_H__

#include "clustbench_types.h"

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


//создание и заполнение файла формата NetCDF
int create_netcdf_header
(
	int file_data_type,
	clustbench_benchmark_parameters_t* test_parameters,
	int *file_id,
	int *data_id,
    int (*benchmark_define_netcdf_vars)(int file_id, clustbench_benchmark_parameters_t* params),
    int (*benchmark_put_netcdf_vars)(int file_id, clustbench_benchmark_parameters_t* params)
);

int create_netcdf_header_3d
(
	int file_data_type,
	clustbench_benchmark_parameters_t *test_parameters,
	int *file_id,
	int *data_id,
    int (*benchmark_define_netcdf_vars)(int file_id, clustbench_benchmark_parameters_t* params),
    int (*benchmark_put_netcdf_vars)(int file_id, clustbench_benchmark_parameters_t* params)
);

//Запись в двумерную матрицу
int netcdf_write_matrix
(
        const int netcdf_file_id,
        const int netcdf_var_id,
        const int matrix_number_in_file,
        const int size_x,
        const int size_y,
        const double *data
);

//Запись в трехмерную матрицу
int netcdf_write_3d_matrix
(
	const int netcdf_file_id,
	const int netcdf_var_id,
	const int matrix_number_in_file,
	const int size_x,
	const int size_y,
	const int size_z,
	const double *data
);

//завершение работы с файлом
int netcdf_close_file(const int netcdf_file_id);

#ifdef __cplusplus
}
#endif

#endif  /* __DATA_WRITE_OPERATIONS_H__ */
