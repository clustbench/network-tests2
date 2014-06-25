#ifndef __FILE_H__
#define	__FILE_H__

#include <stdio.h>
#include "core/data_netcdf.h"

using std::string;

typedef double file_value_t;

///input file description.
typedef struct file_info_t {
    size_t line_length;
    size_t length_min;
    size_t length_step;
    size_t nlines;
} file_info_t;

///Works with input files.
class FileReader
{
public:
    
    void GetFileInfo (file_info_t *file_info_structure) const;
    int ReadLine (file_value_t line []);
    
    //FileReader ();
    FileReader (char *file_name);
    ~FileReader ();
private:
    file_info_t file_info_;
    //FILE *file_stream_;
	data_netcdf *netcdf_ptr_;
	size_t line_;
};

class FileWriter
{
public:
	virtual int WritePartition (size_t line_number, const size_t partition []) = 0;
	
	FileWriter () {};
	virtual ~FileWriter () {};
private:
	FileWriter (const FileWriter&);
};

class StdFileWriter : public FileWriter
{
public:
    
	//int WriteInf (const char *str);
	int WriteLine (const size_t line []);
	
	virtual int WritePartition (size_t line_number, const size_t partition []);
    
    StdFileWriter (const file_info_t &file_info);
    StdFileWriter (char *file_name, const file_info_t &file_info);
    virtual ~StdFileWriter ();
private:
    FILE *file_stream_;
    file_info_t file_info_;
};

#endif	

