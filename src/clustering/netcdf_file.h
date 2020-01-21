#ifndef __NETCDF_FILE_H__
#define __NETCDF_FILE_H__

#include <netcdfcpp.h>
#include "file.h"

/*
 * Data types
 */
#define NUM_NETWORK_TEST_DATATYPES 4

#define AVERAGE_NETWORK_TEST_DATATYPE   1
#define MEDIAN_NETWORK_TEST_DATATYPE    2
#define DEVIATION_NETWORK_TEST_DATATYPE 3
#define MIN_NETWORK_TEST_DATATYPE       4

/*
 * Test types
 */

#define NUM_TEST_TYPES 9

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


class NetCDFFileWriter : public FileWriter
{
public:
	virtual int WritePartition (size_t line_number, const size_t partition []);
	
	NetCDFFileWriter (const char *output_file_name, const char *input_netcdf_name);
	virtual ~NetCDFFileWriter ();
private:
	NcFile *input_;
	NcFile *output_;
	NcVar *data_var_, *input_data_var_;
	size_t line_number_, line_length_, proc_num_;
};

#endif
