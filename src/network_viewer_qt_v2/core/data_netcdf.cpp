#include "data_netcdf.h"
#include <netcdf.h>
#include "data_text.h"

Data_NetCDF::Data_NetCDF (const QString &f_name, const QString &hosts_fname, NV::ErrCode &err) {
	this->source_fname=f_name;
	data_var=-1;

	if (nc_open(f_name.toLocal8Bit().constData()/*written so to process non-English paths*/,
				NC_NOWRITE,&source_file)!=NC_NOERR)
	{
		source_file=-1;
		err=NV::NotANetCDF;
		return;
	}

	int id;
	int tmp_str;

	if ((nc_inq_varid(source_file,"test_type",&id)==NC_NOERR) && 
		(nc_get_var1_int(source_file,id,NULL,&tmp_str)==NC_NOERR))
		this->test_type.setNum(tmp_str);
	if ((nc_inq_varid(source_file,"data_type",&id)==NC_NOERR) && 
		(nc_get_var1_int(source_file,id,NULL,&tmp_str)==NC_NOERR))
		this->data_type.setNum(tmp_str);
	
	#define READVAR(var,var_name,error) \
			if ((nc_inq_varid(source_file,var_name,&id)!=NC_NOERR) || \
				(nc_get_var1(source_file,id,NULL,&var)!=NC_NOERR)) \
			{ err=NV:: error; return; }
	
	READVAR(this->num_processors,"proc_num",NoNumProc)
	READVAR(this->begin_message_length,"begin_mes_length",NoBegMesLen)
	READVAR(this->end_message_length,"end_mes_length",NoEndMesLen)
	READVAR(this->step_length,"step_length",NoStepLen)
	READVAR(this->noise_message_length,"noise_mes_length",NoNoiseMesLen)
	READVAR(this->noise_message_num,"num_noise_mes",NoNoiseMesNum)
	READVAR(this->noise_processors,"num_noise_proc",NoNoiseNumProc)
	READVAR(this->num_repeats,"num_repeates",NoRpts)

	if (!hosts_fname.isEmpty())
	{
		FILE *host_f=fopen(hosts_fname.toLocal8Bit().constData(),"r");
		if (host_f==NULL)
		{
			err=NV::NoHosts;
			return;
		}

		Data_Text::Line l;
		QString name;

		id=1;
		while (Data_Text::readline(host_f,l))
		{
			name.setNum(id);
			(name+=") ")+=l.Give(); // enumerate host's name
			++id;
			this->host_names.push_back(name);
		}
		fclose(host_f);
	}

	int n_vars,i;
	int *varids;

	nc_inq_nvars(source_file,&n_vars);
	varids=static_cast<int*>(malloc(n_vars*sizeof(int)));
	if (varids==NULL)
	{
		err=NV::NoMem;
		return;
	}
	nc_inq_varids(source_file,&n_vars,varids);
	// find first 3D variable - suppose there will be only one
	for (i=0; i<n_vars; ++i)
	{
		nc_inq_varndims(source_file,varids[i],&id);
		if (id==3)
		{
			data_var=varids[i];
			break;
		}
	}
	free(varids);
	if (i==n_vars)
	{
		err=NV::No3DData;
		return;
	}
	
	int dim_id[3];
	size_t len;
	nc_inq_vardimid(source_file,data_var,dim_id);
	nc_inq_dimlen(source_file,dim_id[0],&len);
	if (len==0u)
	{
		err=NV::No3DData;
		return;
	}
	this->z_num=static_cast<int>(len);
	
	start[0]=start[1]=start[2]=len;
	count[0]=count[1]=count[2]=0u;
	
	err=NV::Success;
}

void Data_NetCDF::Begin (const IData::Portion p, const int mes_len) {
	const size_t num_pr=static_cast<size_t>(this->num_processors);
	
	portion=p;
	start[0]=(this->step_length<1)? 0u : static_cast<size_t>((mes_len-this->begin_message_length)/this->step_length);
	start[1]=start[2]=0u;
	switch (p)
	{
		case IData::File: count[0]=static_cast<size_t>(this->z_num)-start[0]; count[1]=count[2]=num_pr; break;
		case IData::Matrix: count[0]=1u; count[1]=count[2]=num_pr; break;
		case IData::Row: count[0]=count[1]=1u; count[2]=num_pr; break;
		case IData::Value: count[0]=count[1]=count[2]=1u; break;
	}
}

NV::ErrCode Data_NetCDF::GetDataAndMove (double *buf) {
	if (start[0]>=static_cast<size_t>(this->z_num)) return NV::InvRead; // Begin() was not called before!
	if (nc_get_vara_double(source_file,data_var,start,count,buf)!=NC_NOERR)
		return NV::ErrNetCDFRead;
	switch (portion)
	{
		case IData::File:
			start[0]=static_cast<size_t>(this->z_num);
			break;
		case IData::Value:
			++(start[2]);
			if (start[2]!=static_cast<size_t>(this->num_processors))
				break;
			start[2]=0u;
			//break;
		case IData::Row:
			++(start[1]);
			if (start[1]!=static_cast<size_t>(this->num_processors))
				break;
			start[1]=0u;
			//break;
		case IData::Matrix:
			++(start[0]);
			break;
	}
	return NV::Success;
}

NV::ErrCode Data_NetCDF::GetSingleValue (const int mes_len, const int row, const int col, double &v)
{
	const size_t st[]={(this->step_length<1)? 0u : 
					   static_cast<size_t>((mes_len-this->begin_message_length)/this->step_length),
					   static_cast<size_t>(row),static_cast<size_t>(col)};
	const size_t cnt[]={1u,1u,1u};
	return ((nc_get_vara_double(source_file,data_var,st,cnt,&v)==NC_NOERR)? NV::Success : NV::ErrNetCDFRead);
}

Data_NetCDF::~Data_NetCDF () {
	if (source_file!=-1) nc_close(source_file);
}

