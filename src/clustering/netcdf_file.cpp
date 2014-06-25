#include <cstdlib>
#include <cstring>


#include "netcdf_file.h"

#define CHECK_THROW(p,s) try {(p);} catch (...) {throw s;}
#define VAR_CREATE_ERR_STR	"NetCDFFileWriter :: NetCDFFileWriter : Error creating variables"
#define VAR_GET_ERR_STR	"NetCDFFileWriter :: NetCDFFileWriter : Error achieving variables"
#define DIM_CREATE_ERR_STR	"NetCDFFileWriter :: NetCDFFileWriter : Error creating dimensions"

NetCDFFileWriter :: NetCDFFileWriter (const char *output_file_name, const char *input_netcdf_name):
input_ (NULL),
output_ (NULL),
data_var_ (NULL),
input_data_var_ (NULL),
line_number_ (0),
line_length_ (0),
proc_num_ (0)
{	
	input_ = new NcFile (input_netcdf_name);
	
	if (!input_->is_valid ()) {
		throw "NetCDFFileWriter :: NetCDFFileWriter : Can't read input file \"" + string (input_netcdf_name) + "\"";
	}
	
	output_ = new NcFile (output_file_name, NcFile :: New);
	
	if (!output_->is_valid ()) {
		throw "NetCDFFileWriter :: NetCDFFileWriter : Can't create file \"" + string (output_file_name) +  "\"";
	}
	
	NcDim *x_dim, *y_dim, *n_dim;
	
	//P_dim = input_->get_dim ("P");
	
	
	
	NcVar *proc_num, *test_type, *data_type, *begin_mes_length, *end_mes_length;
	NcVar *step_length, *noise_mes_length, *num_noise_mes, *num_noise_proc, *num_repeates;
	
	int proc_num_v,	test_type_v, data_type_v, begin_mes_length_v, end_mes_length_v, step_length_v,
	noise_mes_length_v, num_noise_mes_v, num_noise_proc_v, num_repeates_v;
	//int err_count = 0;
	
	//printf ("?");
	
	CHECK_THROW (proc_num_v			= input_->get_var ("proc_num")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (test_type_v		= input_->get_var ("test_type")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (data_type_v		= input_->get_var ("data_type")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (begin_mes_length_v	= input_->get_var ("begin_mes_length")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (end_mes_length_v	= input_->get_var ("end_mes_length")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (step_length_v		= input_->get_var ("step_length")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (noise_mes_length_v	= input_->get_var ("noise_mes_length")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (num_noise_mes_v	= input_->get_var ("num_noise_mes")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (num_noise_proc_v	= input_->get_var ("num_noise_proc")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (num_repeates_v		= input_->get_var ("num_repeates")->as_int (0), VAR_GET_ERR_STR);
	CHECK_THROW (input_data_var_	= input_->get_var ("data"), VAR_GET_ERR_STR);
	
	//proc_num->get (&proc_num_v);
	//test_type->get (&test_type_v);
	//data_type->get (&data_type_v);
	//begin_mes_length->get (&begin_mes_length_v);
	//end_mes_length->get (&end_mes_length_v);
	//step_length->get (&step_length_v);
	//noise_mes_length->get (&noise_mes_length_v);
	//num_noise_mes->get (&num_noise_mes_v);
	//num_noise_proc->get (&num_noise_proc_v);
	//num_repeates->get (&num_repeates_v);
	
	//printf("%d\n", proc_num_v);
	
	CHECK_THROW (x_dim = output_->add_dim ("x", proc_num_v), DIM_CREATE_ERR_STR);
	CHECK_THROW (y_dim = output_->add_dim ("y", proc_num_v), DIM_CREATE_ERR_STR);
	CHECK_THROW (n_dim = output_->add_dim ("n"), DIM_CREATE_ERR_STR);
	
	CHECK_THROW (proc_num			= output_->add_var ("proc_num", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (test_type			= output_->add_var ("test_type", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (data_type			= output_->add_var ("data_type", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (begin_mes_length	= output_->add_var ("begin_mes_length", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (end_mes_length		= output_->add_var ("end_mes_length", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (step_length		= output_->add_var ("step_length", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (noise_mes_length	= output_->add_var ("noise_mes_length", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (num_noise_mes		= output_->add_var ("num_noise_mes", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (num_noise_proc		= output_->add_var ("num_noise_proc", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (num_repeates		= output_->add_var ("num_repeates", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (data_var_			= output_->add_var ("data", ncDouble, n_dim, x_dim, y_dim), VAR_CREATE_ERR_STR);
	
	proc_num->put (&proc_num_v);
	test_type->put (&test_type_v);
	data_type->put (&data_type_v);
	begin_mes_length->put (&begin_mes_length_v);
	end_mes_length->put (&end_mes_length_v);
	step_length->put (&step_length_v);
	noise_mes_length->put (&noise_mes_length_v);
	num_noise_mes->put (&num_noise_mes_v);
	num_noise_proc->put (&num_noise_proc_v);
	num_repeates->put (&num_repeates_v);
	
	line_length_ = proc_num_v*proc_num_v;
	proc_num_ = proc_num_v;
}

NetCDFFileWriter :: ~NetCDFFileWriter ()
{
	if (input_ != NULL) {
		delete input_;
	}
	if (output_ != NULL) {
		delete output_;
	}
}

int NetCDFFileWriter :: WritePartition (size_t line_number, const size_t partition [])
{
	if (line_number <= line_number_) {
		return 1;
	}
	int retval = 0;
	//size_t nlines = line_number - line_number_;
	
	double *means, *values;
	size_t *cluster_sizes, nclusters = 0;
	
	means = new double [line_length_];
	values = new double [line_length_];
	cluster_sizes = new size_t [line_length_];
	memset (cluster_sizes, 0, line_length_*sizeof (size_t));
	
	if (means == NULL || cluster_sizes == NULL || values == NULL) {
		retval = 1;
		goto end;
	}
	
	for (size_t j = 0; j < line_length_; ++j) {
		if (partition [j] > nclusters) {
			nclusters = partition [j];
		}
		++cluster_sizes [partition [j]];
	}
	//printf (">>");
	++nclusters;
	
	for (size_t i = line_number_, j; i < line_number; ++i) {
		memset (means, 0, line_length_*sizeof (double));
		input_data_var_->set_cur(i);
		input_data_var_->get(values, 1, proc_num_, proc_num_);
		
		for (j = 0; j < line_length_; ++j) {
			means [partition [j]] += values [j];
		}
		
		for (j = 0; j < nclusters; ++j) {
			if (cluster_sizes [j] == 0) {
				retval = 2;
				goto end;
			}
			means [j] /= cluster_sizes [j];
		}
		
		for (j = 0; j < line_length_; ++j) {
			values [j] = means [partition [j]];
		}
		
		data_var_->set_cur (i);
		data_var_->put (values, 1, proc_num_, proc_num_);
	}
	//printf ("Everything Allright (%lu)\n", line_number);
	line_number_ = line_number;
	
end:
	if (means != NULL) {
		delete [] means;
	}
	if (values != NULL) {
		delete [] values;
	}
	if (cluster_sizes != NULL) {
		delete [] cluster_sizes;
	}
	if (retval != 0) {
		switch (retval) {
			case 1:
				throw string ("NetCDFFileWriter :: WritePartition : an error has occured");
				break;
			case 2:
				throw string ("NetCDFFileWriter :: WritePartition : critical error - zero size cluster");
				break;
		}
	}
	return 0;
}


