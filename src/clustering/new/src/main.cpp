#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <netcdf.h>
#include "ClusterData.h"
#include "ClusterAbstract.h"
#include "DBScan.h"
#include "Divisive.h"

using namespace std;

double *getData(char *filename, info &a) 
{
	int file = 0, xid = 0, yid = 0, nid = 0;
	int dataid = 0, bmlid = 0, emlid = 0, steplid = 0, pcid = 0;
	size_t x = 0, y = 0, n = 0;
	int status = NC_NOERR;
	status = nc_open(filename, NC_NOWRITE, &file);
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
//TODO:
//параметрики))
int main(int argc, char **argv) 
{
	//-div Nclust medfile devfile N outfilename
	//-dbscan eps clustsize medfile devfile N (optional) outfilename
	//middist mfile dfile N
	int mode, nclust, clustsize, N = -1;
	double eps;
	char *medfile, *devfile, *outfile;
	if (argc == 1) {
		invalidArgs();
		return 0;
	}
	if (string(argv[1]).compare("div") != 0 && string(argv[1]).compare("dbscan") != 0 && (string(argv[1]).compare("avgdist"))) {
		invalidArgs();
		return 0;
	}
	if ((string(argv[1]).compare("avgdist") == 0 && (argc != 4 || argc != 5)) && ((string(argv[1]).compare("dbscan") == 0) && (argc != 7 && argc != 8)) || ((string(argv[1]).compare("div") == 0) && (argc != 6 && argc != 7))) {
		invalidArgs();
		return 0;
	}
	if (string(argv[1]).compare("div") == 0) {
		mode = 1;
		nclust = atoi(argv[2]);
		medfile = argv[3];
		devfile = argv[4];
		outfile = argv[5];
		if (argc == 7)
			N = atoi(argv[6]);
	}
	else if (string(argv[1]).compare("dbscan") == 0) {
		mode = 2;
		eps = atof(argv[2]);
		clustsize = atoi(argv[3]);
		medfile = argv[4];
		devfile = argv[5];
		outfile = argv[6];
		if (argc == 8)
			N = atoi(argv[7]);
	}
	else {
		mode = 3;
		medfile = argv[2];
		devfile = argv[3];
		if (argc == 5)
			N = atoi(argv[4]);
	}
	double *devs, *meds;
	info fileinfo;
	meds = getData(medfile, fileinfo);
	devs = getData(devfile, fileinfo);
	ClustData data;
	data.fitData(meds, devs, fileinfo);
	if (N == -1)
		N = fileinfo.proc_num;
	ClusterAbstract* cl;
	if (mode == 3) {
		double middist = 0;
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				for (int k = 0; k < N; k++)
					for (int m = 0; m < N; m++)
						middist += data.getDist(std::pair <int, int>(i, j), std::pair <int, int>(k, m));
		cout << "Average distance is: " << middist / N*N*N*N << endl;
	} else if (mode == 1) {
		cl = new Divisive(N, nclust, &data);
		std::ofstream ofile(outfile);
		cl->clusterise();
		ofile << "PROC_NUM = " << fileinfo.proc_num << ", BEG_MES_LEN = " << fileinfo.begin_mes_len << ", END_MES_LEN = " << fileinfo.end_mes_len << ", STEP_LEN = " << fileinfo.step_len << endl;
		cl->printData(ofile);
		delete cl;
	}
	else if (mode == 2) {
		cl = new DBScan(N, eps, clustsize, &data);
		std::ofstream ofile(outfile);
		cl->clusterise();
		ofile << "PROC_NUM = " << fileinfo.proc_num << ", BEG_MES_LEN = " << fileinfo.begin_mes_len << ", END_MES_LEN = " << fileinfo.end_mes_len << ", STEP_LEN = " << fileinfo.step_len << endl;
		cl->printData(ofile);
		delete cl;
	}
	delete devs, meds;
	return 0;
}