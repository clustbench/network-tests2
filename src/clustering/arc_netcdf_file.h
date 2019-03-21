#ifndef __ARC_NETCDF_FILE_H__
#define __ARC_NETCDF_FILE_H__

#include <netcdfcpp.h>
#include "file.h"

class ArcNetCDFFileWriter : public FileWriter
{
public:
	virtual int WritePartition (size_t line_number, const size_t partition []);
	
	ArcNetCDFFileWriter (const char *output_file_name, const char *input_netcdf_name);
	virtual ~ArcNetCDFFileWriter ();
private:
	NcFile *input_;
	NcFile *output_, *temp_out_;
	NcVar *data_var_, *input_data_var_, *info_var_, *length_var_;
	size_t line_number_, line_length_, proc_num_, offset_, partition_number_;
	string temp_file_name_;
};

#endif