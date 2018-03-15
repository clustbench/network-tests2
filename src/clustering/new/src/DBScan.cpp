#include "DBScan.h"
#include <iostream>

DBScan::DBScan(int n = 0, double eps = 0, int minPts = 0, ClustData *data = NULL) : eps(eps), minPts(minPts), clustNum(1), data(data) {
	if (n == -1) {
		N = data->getInfo().proc_num;
	}
	else
		N = n;
}

bool DBScan::eps_neighbor(std::pair<int, int> a, std::pair<int, int> b)
{
	return data->getDist(a, b) < eps;
}

std::vector<std::pair<int, int> > DBScan::region_query(std::pair<int, int> el)
{
	std::vector<std::pair<int, int> > result;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			if (eps_neighbor(std::pair<int, int>(i, j), el))
				result.push_back(std::pair<int, int>(i, j));
		}
	return result;
}

bool DBScan::expand_cluster(std::pair<int, int> el)
{
	std::vector<std::pair<int, int> > seeds = region_query(el);
	if (seeds.size() < minPts) {
		classification[el.first][el.second] = NOISE;
		return false;
	}
	else {
		classification[el.first][el.second] = clustNum;
		for (std::vector <std::pair<int, int> >::iterator it = seeds.begin(); it != seeds.end(); ++it) {
			classification[(*it).first][(*it).second] = clustNum;
		}
		while (seeds.size() > 0) {
			std::pair <int, int> cur = seeds[0];
			std::vector<std::pair<int, int> > results;
			if (results.size() >= minPts) {
				for (std::vector <std::pair<int, int> >::iterator it = results.begin(); it != results.end(); ++it)
					if (classification[(*it).first][(*it).second] == UNCLASSIFIED || classification[(*it).first][(*it).second] == NOISE) 
						if (classification[(*it).first][(*it).second] == UNCLASSIFIED)
							seeds.push_back((*it));
			}
			seeds.erase(seeds.begin());
		}
		return true;
	}
}

void DBScan::clusterise()
{
	classification.resize(N);
	for (int i = 0; i < N; i++)
		classification[i].resize(N);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			classification[i][j] = UNCLASSIFIED;
		}
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			if (classification[i][j] == UNCLASSIFIED) {
				if (expand_cluster(std::pair <int, int>(i, j)))
					clustNum++;
			}
		}
	std::vector < std::vector <std::pair<int, int> > > toProceed;
	toProceed.resize(clustNum);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			if (classification[i][j] > 0) 
			toProceed[classification[i][j]].push_back(std::pair<int, int>(i, j));
		}

	for (int i = 0; i < clustNum; i++) {
		result.push_back(Cluster(toProceed[i], 0, 0));
		result[i].calcStats(data);
	}
		
		
}

void DBScan::printData(hid_t file_id) 
{
	std::string name = "/DBSCAN_" + int_to_str(props);
	std::cout << "asdasdasdasd" << std::endl;
	hid_t clustparams_dsp, status;
	hsize_t dims = 1;
   	clustparams_dsp = H5Screate_simple(1, &dims, NULL);

	hid_t dbgroup_id = H5Gcreate(file_id, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hid_t eps_id = H5Acreate(dbgroup_id, "EPS", H5T_IEEE_F64BE, clustparams_dsp, H5P_DEFAULT, H5P_DEFAULT);
	hid_t minPts_id = H5Acreate(dbgroup_id, "MINPTS", H5T_STD_I32BE, clustparams_dsp, H5P_DEFAULT, H5P_DEFAULT);
	std::cout << minPts << " " << eps << std::endl;
	status = H5Awrite(eps_id,  H5T_NATIVE_DOUBLE, &eps);
	status = H5Awrite(minPts_id, H5T_NATIVE_INT, &minPts);
	H5Sclose(clustparams_dsp);
	H5Aclose(eps_id);
	H5Aclose(minPts_id);
	for (int i = 0; i < result.size(); i++) {
		std::string clustname  = "/CLUSTER_" + int_to_str(i);
		hid_t clustgroupid = H5Gcreate(dbgroup_id, clustname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		result[i].printData(clustgroupid);
		std::cout << clustgroupid << std::endl;
		H5Gclose(clustgroupid);
			
	}
	status = H5Gclose(dbgroup_id);
	props++;

}


DBScan::~DBScan(){}
