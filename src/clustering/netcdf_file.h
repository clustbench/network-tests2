#ifndef __NETCDF_FILE_H__
#define __NETCDF_FILE_H__

#include <netcdfcpp.h>
#include "file.h"

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