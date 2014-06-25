#include "file.h"

void FileReader :: GetFileInfo(file_info_t *file_info_structure) const {
    file_info_structure->length_min = file_info_.length_min;
    file_info_structure->length_step = file_info_.length_step;
    file_info_structure->line_length = file_info_.line_length;
    file_info_structure->nlines = file_info_.nlines;
}

int FileReader :: ReadLine(file_value_t line []) {
	if (netcdf_ptr_ == NULL) {
		/*M33N5 32202*/
		return 0;
	}
	if (line_ >= file_info_.nlines) {
		/*That's all*/
		return 0;
	}
	/*In netcdf_data.h there is an error here. Cause it devides length by step_length (what an 
	 *awfull work: multiply by step_length, then divide. Also there is a big problem: this lib 
	 *needs matrix of double values, in getMatrix function this matrix is created, then (!) 
	 *it's converted the list of lists (!) and after that it's converted back to linear matrix 
	 *of double values.) and not consider min_length. Everything works pretty well before 
	 *min_length becomes greater than step_length. Drama. But also a hack - this library divides 
	 *by step_length to achieve line number, so we'll feed it with line number we want multiplied
	 *by length_step.
	 *And the most bad thing - this function doesn't check input value and this value is
	 *of integer type! Negative length is so wonderful.
	 */
	data_netcdf :: matrix matrix = netcdf_ptr_->getMatrix (line_ * file_info_.length_step);
	
	data_netcdf :: matrix :: iterator it1;
	/*Another bad thing: to get needed iterator type we must know about matrix structure, so
	 *attempt of data encapsulation dies.
	 */
	list <double> :: iterator it2;
	size_t i;
	for (it1 = matrix.begin (), i = 0; it1 != matrix.end (); ++it1) {
		for (it2 = it1->begin (); it2 != it1->end (); ++it2, ++i) {
			line [i] = *it2;
		}
	}
	++line_;
	return 1;
}

/*int FileReader :: ReadLine(file_value_t line []) {
    int flag = 0;
    for (size_t i = 0; i < file_info_.line_length; ++i) {
        flag = fscanf(file_stream_, "%lg", line + i);
        if (flag == 0) {
            return 0;
        }
    }
    return 1;
}*/

/*FileReader :: FileReader() {
    file_stream_ = stdin;
    
    file_info_.length_min = 0;
    file_info_.length_step = 100;
    file_info_.line_length = 16;
    file_info_.nlines = 10;
}*/

FileReader :: FileReader(char *file_name) {
	try {
		netcdf_ptr_ = new data_netcdf (file_name);
	}
	catch (string err_string) {
		//printf ("%s\n That means it is wrong file format \n", err_string);
		throw "FileReader::FileReader : Can't read file \"" + string (file_name) + "\"\n"+err_string;
	}
	
	file_info_.length_min = netcdf_ptr_->getBeginMessageLength ();
	file_info_.length_step = netcdf_ptr_->getStepLength ();
	file_info_.line_length = netcdf_ptr_->getNumProcessors ()*netcdf_ptr_->getNumProcessors ();
	file_info_.nlines = (netcdf_ptr_->getRealEndMessageLength () - netcdf_ptr_->getBeginMessageLength ()) / netcdf_ptr_->getStepLength ();
	line_ = 0;
}


FileReader :: ~FileReader () {
	if (netcdf_ptr_ != NULL) {
		delete netcdf_ptr_;
	}
}
//FileReader :: ~FileReader() {
//    if (file_stream_ != NULL && file_stream_ != stdin) {
//        fclose(file_stream_);
//    }
//}
/******************************************************************************/

//int FileStdWriter :: WriteInf (const char *str) {
//	fprintf(file_stream_, "%s\n", str);
//}

int StdFileWriter :: WriteLine(const size_t line []) {
	for (size_t i = 0; i < file_info_.line_length; ++i) {
		fprintf(file_stream_, "%lu ", line [i]);
	}
	fprintf (file_stream_, "\n");
	return 0;
}

int StdFileWriter :: WritePartition(size_t line_number, const size_t partition [])
{
	fprintf (file_stream_, "#%lu\n", line_number);
	WriteLine (partition);
	return 0;
}

StdFileWriter :: StdFileWriter(const file_info_t &file_info):
file_info_(file_info) {
    file_stream_ = stdout;
}

StdFileWriter :: StdFileWriter(char *file_name, const file_info_t &file_info):
file_info_(file_info) {
    
}

StdFileWriter :: ~StdFileWriter() {
    if (file_stream_ != NULL && file_stream_ != stdout) {
        fclose(file_stream_);
    }
}
