#pragma once
#include <vector>
#include <utility>
#include <fstream>
#include <cmath>
#include <string>
#include <iostream>
#include <limits>
#include "hdf5/serial/hdf5.h"
#include <cstdlib>

class ClustData;

struct info {
	int begin_mes_len, end_mes_len, step_len, proc_num;
	size_t sz[3];
};

class Cluster 
{
public:
	std::pair <Cluster*, Cluster*> childs;
	std::vector <double> m, d;
	Cluster* father;
	std::vector <std::pair<int, int> > elements;
	std::pair <int, int> seed;
public:
	Cluster();
	Cluster(std::vector <std::pair <int, int> >, int, int);

	void calcStats(ClustData*);
	void printData(hid_t);

	void setChilds(std::pair <Cluster*, Cluster*>);
	void setFather(Cluster*);
	void setElements(std::vector <std::pair<int, int> >);
	void setSeed(std::pair <int, int>);

	bool isHollow();

	std::pair <Cluster*, Cluster*> getChilds();
	Cluster *getFather();
	std::vector <std::pair<int, int> > getElements();
	std::pair <int, int> getSeed();
};

class ClustData 
{
public:
	typedef struct elem {
		double *med;
		double *dev;
	} elem;
	double getDist(std::pair<int, int> , std::pair<int, int>);
	void fitData(double*, double*, info);
	info getInfo() {
		return fileinfo;
	}
	double getMed(std::pair<int, int>, int);
private:
	std::vector <std::vector <elem> > clData;
	info fileinfo;
};

std::string int_to_str(int);