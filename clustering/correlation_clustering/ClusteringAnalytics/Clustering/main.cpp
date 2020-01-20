#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <netcdf.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "Statistics.h"
#include "ClusterData.h"
#include "ClusterAbstract.h"
#include "DBScan.h"
#include "KMeans.h"

using namespace std;

double threshold = 0.0;

double *getData(const char *filename, info &a) 
{
	int file = 0, xid = 0, yid = 0, nid = 0;
	int dataid = 0, bmlid = 0, emlid = 0, steplid = 0, pcid = 0;
	size_t x = 0, y = 0, n = 0;
	int status = NC_NOERR;
	status = nc_open(filename, NC_NOWRITE, &file);
	size_t count[3] = { 0 , 0, 0 };
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

void save3D(vector<vector<vector<double>>> const &data, hid_t id, const char* name) {
	unsigned int N = data.size();
	unsigned int L = data[0][0].size();
	hsize_t dims[3] = { N, N, L };
	double *corr_data = new double[N * N * L];
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < L; k++)
				corr_data[i * N * L + j * L + k] = data[i][j][k];
	H5LTmake_dataset_double(id, "CORR", 3, dims, corr_data);
	H5LTset_attribute_string(id, "/", "TYPE", name);
	delete[] corr_data;
}

void invalidArgs() 
{
	cout << "Usage:" << endl;
	//cout << "For Divisive clusterisation with nclust number of clusters:" << endl;
	//cout << "div nclust medfilename devfilename outfilename [first_proc_num]" << endl;
	//cout << "For DBScan clusterisation with eps and clustsize parameters:" << endl;
	//cout << "dbscan eps clustsize medfile devfile outfilename [first_proc_num]" << endl;
}
//db 3 0.5 3 file
ProgParams parseArgs(int argc, char **argv)
{
	ProgParams result;
	result.cluteringOptions.type = -1;
	if (string(argv[1]) == "kmeans") {
		result.cluteringOptions.type = 1;
	} else if (string(argv[1]) == "dbscan") {
		result.cluteringOptions.type = 2;
	}
	else if (string(argv[1]) == "peaks") {
		result.cluteringOptions.type = 3;
	}
	else if (string(argv[1]) == "jumps") {
		result.cluteringOptions.type = 4;
	} else {
		invalidArgs();
		result.num_proc = -1;
		return result;
	}
	switch (result.cluteringOptions.type) {
	case 1:
		result.cluteringOptions.k = atoi(argv[2]);
		threshold = atof(argv[3]);
		result.medFile_name = string(argv[4]);
		result.devFile_name = "";
		result.outputFile_name = string(argv[5]);
		result.num_proc = 1;
		break;
	case 2:
		result.cluteringOptions.eps = atof(argv[2]);
		result.cluteringOptions.n_clust = atoi(argv[3]);
		result.medFile_name = string(argv[5]);
		result.devFile_name = string(argv[6]);
		result.outputFile_name = string(argv[7]);
		result.num_proc = atoi(argv[4]);
		break;
	case 3:
	case 4:
		result.medFile_name = string(argv[2]);
		result.outputFile_name = string(argv[3]);
		break;
	default:
		invalidArgs();
		result.num_proc = -1;
	}
	return result;
}

int main(int argc, char **argv) 
{
	ProgParams params = parseArgs(argc, argv);
	if (params.num_proc == -1) {
		return -1;
	}
	double *devs = NULL, *meds = NULL;
	info fileinfo;
	meds = getData(params.medFile_name.c_str(), fileinfo);
	if (!params.devFile_name.empty())
		devs = getData(params.devFile_name.c_str(), fileinfo);
	if (params.num_proc == -1) {
		return -1;
	}
	hid_t outputFile_id = H5Fcreate(params.outputFile_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (outputFile_id < 0) {
		std::cout << "Something went wrong during file creation (file is opened somewhere else?)" << std::endl;
		return -2;
	}
	hid_t status;
	//hid_t fileinfo_dsp;
	//hsize_t dims = 1;
	//fileinfo_dsp = H5Screate_simple(1, &dims, NULL);
	H5LTset_attribute_int(outputFile_id, "/", "NUM_PROC", &fileinfo.proc_num, 1);
	H5LTset_attribute_int(outputFile_id, "/", "BEG_MES_LEN", &fileinfo.begin_mes_len, 1);
	H5LTset_attribute_int(outputFile_id, "/", "END_MES_LEN", &fileinfo.end_mes_len, 1);
	H5LTset_attribute_int(outputFile_id, "/", "STEP", &fileinfo.step_len, 1);

	ClustData readData;
	readData.fitData(meds, devs, fileinfo);
	if (params.cluteringOptions.type == 1) {
		KMeans kmeans(params.cluteringOptions.k, readData, threshold);
		kmeans.clusterise();
		kmeans.convert(readData);
		kmeans.printData(outputFile_id);
	}
	else if (params.cluteringOptions.type == 2) {
		DBScan db(params.num_proc, params.cluteringOptions.eps, params.cluteringOptions.n_clust, readData);
		db.clusterise();
		db.printData(outputFile_id);
	}
	else if (params.cluteringOptions.type == 3) {
		vector <vector <vector <double>>> corr_p = corr_peaks(readData);
		save3D(corr_p, outputFile_id, "CORR_PEAKS");
	}
	else if (params.cluteringOptions.type == 4) {
		vector <vector <vector <double>>> corr_j = corr_jumps(readData);
		save3D(corr_j, outputFile_id, "CORR_JUMPS");
	}

	H5Fclose(outputFile_id);

	return 0;
}