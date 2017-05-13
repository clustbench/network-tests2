#pragma once
#include "ClusterData.h"
#include "ClusterAbstract.h"
#include <vector>
#include <utility>
#include <fstream>

/*
DBScan Clustering
*/

class DBScan : public ClusterAbstract
{
private:
	enum {
		UNCLASSIFIED = -2, NOISE = -1
	};
	ClustData *data;
	double eps;
	int minPts;
	int clustNum, N;
	std::vector < std::vector <int> > classification;
	std::vector < Cluster > result;


	std::vector <std::pair <int, int> > region_query(std::pair<int, int>);
	bool expand_cluster(std::pair<int, int>);
	bool eps_neighbor(std::pair <int, int>, std::pair<int, int>);

public:
	DBScan(int, double, int, ClustData*);

	void clusterise(); //main method
	void printData(std::ofstream&);

	~DBScan();

};

