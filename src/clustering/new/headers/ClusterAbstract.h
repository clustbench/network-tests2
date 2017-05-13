#pragma once
#include <fstream>

class ClusterAbstract
{
public:
	ClusterAbstract();
	virtual void clusterise() = 0;
	virtual void printData(std::ofstream&) = 0;
	virtual ~ClusterAbstract();
};

