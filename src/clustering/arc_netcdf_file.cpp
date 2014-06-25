#include <cstdlib>
#include <cstring>

#include "arc_netcdf_file.h"

#define CHECK_THROW(p,s) try {(p);} catch (...) {throw s;}
#define VAR_CREATE_ERR_STR	"ArcNetCDFFileWriter :: ArcNetCDFFileWriter : Error creating variables"
#define VAR_GET_ERR_STR	"ArcNetCDFFileWriter :: ArcNetCDFFileWriter : Error achieving variables"
#define DIM_GET_ERR_STR	"ArcNetCDFFileWriter :: ArcNetCDFFileWriter : Error achieving dimensions"
#define DIM_CREATE_ERR_STR	"ArcNetCDFFileWriter :: ArcNetCDFFileWriter : Error creating dimensions"

ArcNetCDFFileWriter :: ArcNetCDFFileWriter (const char *output_file_name, const char *input_netcdf_name):
input_ (NULL),
output_ (NULL),
temp_out_ (NULL),
data_var_ (NULL),
input_data_var_ (NULL),
info_var_ (NULL),
length_var_ (NULL),
line_number_ (0),
line_length_ (0),
proc_num_ (0),
offset_ (0),
partition_number_ (0)
{	
	input_ = new NcFile (input_netcdf_name);
	
	if (!input_->is_valid ()) {
		throw "NetCDFFileWriter :: NetCDFFileWriter : Can't read input file \"" + string (input_netcdf_name) + "\"";
	}
	
	output_ = new NcFile (output_file_name, NcFile :: New);
	
	if (!output_->is_valid ()) {
		throw "NetCDFFileWriter :: NetCDFFileWriter : Can't create file \"" + string (output_file_name) +  "\"";
	}
	
	temp_file_name_ = string (output_file_name) + ".!nc";
	
	temp_out_ = new NcFile (temp_file_name_.data (), NcFile :: New);
	
	if (!temp_out_->is_valid ()) {
		throw "NetCDFFileWriter :: NetCDFFileWriter : Can't create file \"" + string (output_file_name) +  "\"";
	}
	
	NcDim *x_dim, *i_dim, *n_dim, *d_dim;
	
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
	
	CHECK_THROW (n_dim = input_->get_dim ("n"), DIM_GET_ERR_STR);
	
	CHECK_THROW (x_dim = output_->add_dim ("ax", proc_num_v), DIM_CREATE_ERR_STR);
	CHECK_THROW (i_dim = output_->add_dim ("ai", n_dim->size ()), DIM_CREATE_ERR_STR);
	CHECK_THROW (n_dim = output_->add_dim ("an"), DIM_CREATE_ERR_STR);
	CHECK_THROW (d_dim = temp_out_->add_dim ("ad"), DIM_CREATE_ERR_STR);
	
	CHECK_THROW (proc_num			= output_->add_var ("aproc_num", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (test_type			= output_->add_var ("atest_type", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (data_type			= output_->add_var ("adata_type", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (begin_mes_length	= output_->add_var ("abegin_mes_length", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (end_mes_length		= output_->add_var ("aend_mes_length", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (step_length		= output_->add_var ("astep_length", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (noise_mes_length	= output_->add_var ("anoise_mes_length", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (num_noise_mes		= output_->add_var ("anum_noise_mes", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (num_noise_proc		= output_->add_var ("anum_noise_proc", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (num_repeates		= output_->add_var ("anum_repeates", ncInt), VAR_CREATE_ERR_STR);
	CHECK_THROW (info_var_			= output_->add_var ("ainfo", ncInt, n_dim, x_dim, x_dim), VAR_CREATE_ERR_STR);
	CHECK_THROW (length_var_		= output_->add_var ("alength", ncInt, n_dim), VAR_CREATE_ERR_STR);
	CHECK_THROW (data_var_			= temp_out_->add_var ("data", ncDouble, d_dim), VAR_CREATE_ERR_STR);
	
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

ArcNetCDFFileWriter :: ~ArcNetCDFFileWriter ()
{
	int retval = 0;
	string err_str;
	if (input_ != NULL) {
		delete input_;
	}
	if (output_ != NULL && temp_out_ != NULL) {
		NcDim *d_dim;
		NcVar *new_data_var;
		size_t data_size = 0;
		try {
			CHECK_THROW (d_dim = temp_out_->get_dim ("ad"), DIM_GET_ERR_STR);
			data_size = d_dim->size ();

			CHECK_THROW (d_dim = output_->add_dim ("ad", data_size), DIM_CREATE_ERR_STR);
			CHECK_THROW (new_data_var = output_->add_var ("adata", ncDouble, d_dim), VAR_CREATE_ERR_STR);
			double *data = new double [data_size];
			if (data == NULL) {
				throw string ("ArcNetCDFFileWriter :: ~ArcNetCDFFileWriter : Not enough memory");
			}

			data_var_->set_cur();
			data_var_->get (data, data_size);
			
			new_data_var->set_cur();
			new_data_var->put (data, data_size);
			
			delete [] data;
		} catch (string str) {
			retval = 1;
			err_str = str;
		}
	}
	if (output_ != NULL) {
		delete output_;
	}
	if (temp_out_ != NULL) {
		delete temp_out_;
		remove (temp_file_name_.data ());
	}
	if (retval != 0) {
		throw err_str;
	}
}

int ArcNetCDFFileWriter :: WritePartition (size_t line_number, const size_t partition [])
{
	if (line_number <= line_number_) {
		return 1;
	}
	int retval = 0;
	size_t nlines = line_number - line_number_;
	
	double *means, *values;
	int *partition_offset, lnum;
	size_t *cluster_sizes, nclusters = 0;
	
	
	partition_offset = new int [line_length_];
	means = new double [line_length_];
	values = new double [line_length_];
	cluster_sizes = new size_t [line_length_];
	memset (cluster_sizes, 0, line_length_*sizeof (size_t));
	
	if (means == NULL || cluster_sizes == NULL || values == NULL || partition_offset == NULL) {
		retval = 1;
		goto end;
	}
	
	for (size_t j = 0; j < line_length_; ++j) {
		if (partition [j] > nclusters) {
			nclusters = partition [j];
		}
		++cluster_sizes [partition [j]];
	}
	++nclusters;
	
	for (size_t j = 0; j < line_length_; ++j) {
		partition_offset [j] = partition [j]*nlines + offset_;
	}
	
	info_var_->set_cur (partition_number_);
	info_var_->put (partition_offset, 1, proc_num_, proc_num_);
	length_var_->set_cur(partition_number_);
	lnum = (long int)line_number;
	length_var_->put (&(lnum), 1);
	
	for (size_t i = line_number_, j; i < line_number; ++i) {
		memset (means, 0, line_length_*sizeof (double));
		input_data_var_->set_cur(i);
		input_data_var_->get(values, 1, proc_num_, proc_num_);
		
		for (j = 0; j < line_length_; ++j) {
			means [partition [j]] += values [j];
		}
		for (j = 0; j < nclusters; ++j) {
			means [j] /= cluster_sizes [j];
			data_var_->set_cur (j*nlines + offset_ + i);
			data_var_->put (means+j, 1);
		}
		
		//data_var_->set_cur (i);
		//data_var_->put (values, 1, proc_num_, proc_num_);
	}
	//printf ("Everything Allright (%lu)\n", line_number);
	
	++partition_number_;
	offset_ += nlines*nclusters;
	line_number_ = line_number;
end:
	if (partition_offset != NULL) {
		delete [] partition_offset;
	}
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
				throw string ("ArcNetCDFFileWriter :: WritePartition : an error has occured");
				break;
			case 2:
				throw string ("ArcNetCDFFileWriter :: WritePartition : critical error - zero size cluster");
				break;
		}
	}
	return 0;
}


