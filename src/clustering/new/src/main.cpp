#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <netcdf.h>
#include "ClusterData.h"
#include "ClusterAbstract.h"
#include "DBScan.h"
#include "Divisive.h"
#include "ConfRoutine.h"

using namespace std;

int DBScan::props = 0;

double *getData(const char *filename, info &a) 
{
	int file = 0, xid = 0, yid = 0, nid = 0;
	int dataid = 0, bmlid = 0, emlid = 0, steplid = 0, pcid = 0;
	size_t x = 0, y = 0, n = 0;
	int status = NC_NOERR;
	status = nc_open(filename, NC_NOWRITE, &file);
	if (status != NC_NOERR) {
		std::cout << "File " << filename << " wasn't found" << std::endl;
		return NULL;
	}
	size_t count[3];
	double *data = NULL;
	if (status == NC_NOERR) {
		//preparations
		nc_inq_dimid(file, "x", &xid); nc_inq_dimid(file, "y", &yid); nc_inq_dimid(file, "n", &nid);
		nc_inq_dimlen(file, xid, &x); nc_inq_dimlen(file, yid, &y); nc_inq_dimlen(file, nid, &n);
		nc_inq_varid(file, "data", &dataid);
		nc_inq_varid(file, "begin_mes_length", &bmlid);
		nc_inq_varid(file, "end_mes_length", &emlid);
		nc_inq_varid(file, "step_length", &steplid);
		nc_inq_varid(file, "proc_num", &pcid);
		//get data
		data = new double[x*y*n];
		size_t start[3] = { 0, 0, 0 };
		count[0] =  n, count[1] = x, count[2]= y;
		nc_get_vara_double(file, dataid, start, count, data);
		nc_get_var_int(file, bmlid, &a.begin_mes_len);
		nc_get_var_int(file, emlid, &a.end_mes_len);
		nc_get_var_int(file, steplid, &a.step_len);
		nc_get_var_int(file, pcid, &a.proc_num);
	}
	a.sz[0] = count[0];
	a.sz[1] = count[1];
	a.sz[2] = count[2];
	return data;
}

void invalidArgs() 
{
	cout << "Usage:" << endl;
	cout << "For Divisive clusterisation with nclust number of clusters:" << endl;
	cout << "div nclust medfilename devfilename outfilename [first_proc_num]" << endl;
	cout << "For DBScan clusterisation with eps and clustsize parameters:" << endl;
	cout << "dbscan eps clustsize medfile devfile outfilename [first_proc_num]" << endl;
}

int main(int argc, char **argv) 
{
	std::string confname = "params.conf";
	if (argc != 1) {
		confname = std::string(argv[1]);
	}
	ConfParser default_parser = ConfParser(confname);
	int st = default_parser.parse_file();
	if (st < 0)
		return st;
	ProgParams params;
	st = default_parser.calc_params(params);
	if (st < 0)
		return st;
	for (int i = 0; i < params.clusteringOptions.size(); i++) {
		std::cout << params.clusteringOptions[i].name << std::endl;
		for (int j = 0; j < params.clusteringOptions[i].float_parameters.size(); j++) {
			std::cout << params.clusteringOptions[i].float_parameters[j] << " ";

		}
		std::cout << std::endl;

		for (int j = 0; j < params.clusteringOptions[i].int_parameters.size(); j++) {
			std::cout << params.clusteringOptions[i].int_parameters[j] << " ";

		}
		std::cout << std::endl;

	}
	hid_t outputFile_id = H5Fcreate(params.outputFile_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (outputFile_id < 0) {
		std::cout << "Something went wrong during file creation (file is opened somewhere else?)" << std::endl;
		return -2;
	}

	hid_t status;
	hid_t fileinfo_dsp;
	hsize_t dims = 1;
   	fileinfo_dsp = H5Screate_simple(1, &dims, NULL);
//TODO:
//names of nodes.txt
	hid_t procNum_attr = H5Acreate(outputFile_id, "PROC_NUM", H5T_STD_I32BE, fileinfo_dsp, H5P_DEFAULT, H5P_DEFAULT);
	hid_t begMesLen_attr = H5Acreate(outputFile_id, "BEG_MES_LEN", H5T_STD_I32BE, fileinfo_dsp, H5P_DEFAULT, H5P_DEFAULT);
	hid_t endMesLen_attr = H5Acreate(outputFile_id, "END_MES_LEN", H5T_STD_I32BE, fileinfo_dsp, H5P_DEFAULT, H5P_DEFAULT);
	hid_t stepMesLen_attr = H5Acreate(outputFile_id, "STEP", H5T_STD_I32BE, fileinfo_dsp, H5P_DEFAULT, H5P_DEFAULT);
//begin_mes_len, end_mes_len, step_len, proc_num;
	double *devs, *meds;
	info fileinfo;
	meds = getData(params.medFile_name.c_str(), fileinfo);
	if (meds == NULL)
		return -1;
	devs = getData(params.devFile_name.c_str(), fileinfo);
	if (devs == NULL) {
		delete meds;
		return -1;
	}
	ClustData data;
	data.fitData(meds, devs, fileinfo);
	if (params.num_proc > 0)
		status = H5Awrite(procNum_attr, H5T_NATIVE_INT, &params.num_proc);
	else
		status = H5Awrite(procNum_attr, H5T_NATIVE_INT, &fileinfo.proc_num);
	status = H5Awrite(begMesLen_attr, H5T_NATIVE_INT, &fileinfo.begin_mes_len);
	status = H5Awrite(endMesLen_attr, H5T_NATIVE_INT, &fileinfo.end_mes_len);
	status = H5Awrite(stepMesLen_attr, H5T_NATIVE_INT, &fileinfo.step_len);
	H5Aclose(procNum_attr);
	H5Aclose(begMesLen_attr);
	H5Aclose(endMesLen_attr);
	H5Aclose(stepMesLen_attr);
	H5Sclose(fileinfo_dsp);
	int np = params.num_proc > 0 ? params.num_proc : fileinfo.proc_num;
	for (int i = 0; i < params.clusteringOptions.size(); i++) {
		if (params.clusteringOptions[i].name == "dbscan") {
			for (int j = 0; j < params.clusteringOptions[i].float_parameters.size(); j++) {
				ClusterAbstract* cl = new DBScan(np, params.clusteringOptions[i].float_parameters[j], params.clusteringOptions[i].int_parameters[j], &data);
				cl->clusterise();
				std::cout << "1231231" << std::endl;
				cl->printData(outputFile_id);
				delete cl;

			}
		}
	}
	//	if (params.clusteringOptions[i].name == "div") {
	//		for (int j = 0; j < params.clusteringOptions[i].float_parameters.size(); j++) {
	//			ClusterAbstract* cl = new Divisive(params.clusteringOptions[i].int_parameters[j], params.num_proc, data);
	//			cl->clusterise();
	//		//	cl->printData();
	//			delete cl;

	//		}
	//	}
	//}


	H5Fclose(outputFile_id);
	delete devs, meds;
	return 0;
}
