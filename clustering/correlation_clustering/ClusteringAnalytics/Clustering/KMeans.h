#pragma once
#include <vector>
#include <iostream>
#include <utility>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "ClusterAbstract.h"
#include "ClusterData.h"
#include <hdf5.h>
#include <hdf5_hl.h>
#include "Statistics.h"


using namespace std;

class KMeans : public ClusterAbstract
{
public:
	KMeans(int K, ClustData clData, double);
	void clusterise();
	void KMeans::convert(ClustData &raw_vals);
	void printData(hid_t outputFile_id);
private:
	double threshold;
	vector <vector <vector <double>>> features;
	vector <vector <double>> centroids;
	vector <vector <int>> labels;
	vector < vector <double>> recalculate_centroids();
	std::vector<std::vector<ClustData::elem>> cl_data;
	
	void initialize();
	double calc_distance(int, int, vector<double>);
	vector <Cluster> cluster_result;
	bool check_dissim(vector <pair<int, int>> a);
	bool check_dissim(vector <vector <double>> const &a, vector <vector <double>> const &b);
	int num_proc, mes_total;
	int K;
	void calc_features();
};

