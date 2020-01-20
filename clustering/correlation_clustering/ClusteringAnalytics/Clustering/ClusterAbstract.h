#pragma once
#include <fstream>
#include <vector>
#include <string>
#include "hdf5.h"

class ClusterAbstract
{
public:
	ClusterAbstract();
	virtual void clusterise() = 0;
	virtual void printData(hid_t) = 0;
	virtual ~ClusterAbstract();
};

struct ProgParams
{
	struct ClusterParams {
		double eps = 0, n_clust = 0;
		int k = 0;
		int type;
	};
	int num_proc;
	ClusterParams cluteringOptions;
	std::string medFile_name;
	std::string devFile_name;
	std::string outputFile_name;
};