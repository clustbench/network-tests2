#ifndef DATA_CLUST_H
#define DATA_CLUST_H
#include <utility>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>

class Cluster
{
public:
    Cluster(std::vector < std::pair <int, int> > inData, std::vector < double > inM, std::vector < double> inD) : data(inData), m(inM), d(inD) {}

    std::vector < std::pair <int, int> > getData();
    std::vector < double > getM();
    std::vector < double > getD();

private:
    std::vector < std::pair <int, int> > data;
    std::vector < double > m, d;
};

class ClusterReader
{
public:
    ClusterReader(std::string);

    void readFromFile();

    int getProcNum();
    int getBegMesLen();
    int getEndMesLen();
    int getStepLen();

    std::vector < Cluster > getClusters();

private:
    std::vector < Cluster > clusters;
    std::string filename;
    int proc_num, beg_mes_len, end_mes_len, step_len;
};

#endif // DATA_CLUST_H
