#pragma once
#include <vector>
#include <utility>
#include <fstream>
#include <hdf5.h>
#include <hdf5_hl.h>

using namespace std;

class ClustData;

struct info {
	int begin_mes_len, end_mes_len, step_len, proc_num;
	size_t sz[3];
};

std::string int_to_str(int a);

class Cluster 
{
public:
	std::vector <double> centroid;
	std::vector <int> features;
	std::vector <std::pair<int, int> > elements;

	Cluster();
	Cluster(std::vector <std::pair <int, int> >, int, int);

	void printData(hid_t);

	bool isHollow();

};

class ClustData 
{
public:
	typedef struct elem {
		std::vector <double> med;
		std::vector <double> dev;
	} elem;
	double getDist(std::pair<int, int> , std::pair<int, int>);
	void fitData(double*, double*, info);
	info getInfo() {
		return fileinfo;
	}
	std::vector <std::vector <elem> > getClData();
	double getMed(std::pair<int, int>, int);
	std::vector <double> get_med_series(std::pair<int, int> a);
private:
	std::vector <std::vector <elem> > clData;
	info fileinfo;
};

