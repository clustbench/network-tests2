#include "DBScan.h"

DBScan::DBScan(int n = 0, double eps = 0, int minPts = 0, ClustData &data = ClustData()) : eps(eps), minPts(minPts), clustNum(1), data(data) {
	if (n == -1)
		N = data.getInfo().proc_num;
	else
		N = n;
}

bool DBScan::eps_neighbor(std::pair<int, int> a, std::pair<int, int> b)
{
	return data.getDist(a, b) < eps;
}

std::vector<std::pair<int, int> > DBScan::region_query(std::pair<int, int> el)
{
	info a = data.getInfo();
	std::vector<std::pair<int, int> > result;
	for (int i = 0; i < a.proc_num; i++)
		for (int j = 0; j < a.proc_num; j++) {
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
	result.resize(clustNum);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			toProceed[classification[i][j]].push_back(std::pair<int, int>(i, j));
		}
	for (int i = 0; i < clustNum; i++) {
		result[i]->elements = toProceed[i];
	 //	result[i]->calcStats(data);
	}
		
}

void DBScan::printData(hid_t outFile_id) 
{
	H5LTset_attribute_string(outFile_id, "/", "CLUSTERING_TYPE", "DBSCAN");
	H5LTset_attribute_double(outFile_id, "/", "EPS", &eps, 1);
	H5LTset_attribute_int(outFile_id, "/", "minPts", &minPts, 1);
	for (int i = 0; i < result.size(); i++) {
		std::string clustname = "/CLUSTER_" + int_to_str(i);
		hid_t clustgroupid = H5Gcreate(outFile_id, clustname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		result[i]->printData(clustgroupid);
		H5Gclose(clustgroupid);
	}
}


DBScan::~DBScan(){}
