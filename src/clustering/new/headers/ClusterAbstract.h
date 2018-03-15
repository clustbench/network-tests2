#pragma once
#include <fstream>
#include <vector>
#include "hdf5/serial/hdf5.h"

class ClusterAbstract
{
public:
	ClusterAbstract();
	virtual void clusterise() = 0;
	virtual void printData(hid_t) = 0;
	virtual ~ClusterAbstract();
};
