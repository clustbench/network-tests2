#pragma once
#include "ClusterData.h"
#include "ClusterAbstract.h"
#include <vector>
#include <list>
#include <utility>
#include <cstdlib>
#include <ctime>

class Divisive : public ClusterAbstract
{
private:
	int N, totalClusters, estClustNum;
	Cluster *head;
	ClustData *dataset;
	std::pair <int, int> findFarest(int, int, Cluster*);
	void divideCluster(Cluster*);
	std::vector <Cluster*> result;
public:
	Divisive(int, int, ClustData*);

	void clusterise();
	void printData(std::ofstream&);

	~Divisive();

};

