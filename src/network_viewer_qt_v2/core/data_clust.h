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
    Cluster();
private:
    std::vector < std::pair <int, int> > data;
    std::vector < double > m, d;
};

class ClusterReader
{
public:
    ClusterReader(char*);
    void readFromFile();
private:
  //  std::vector < Cluster > clusters;
    char *filename;
    int proc_num, max_mes_len, mes_len_step;
};

#endif // DATA_CLUST_H
