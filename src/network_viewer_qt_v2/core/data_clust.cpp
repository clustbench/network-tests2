#include "data_clust.h"

#include <iostream>
ClusterReader::ClusterReader(char *fname = NULL) : filename(fname)
{
}

void ClusterReader::readFromFile()
{
    std::ifstream infile(filename);
    char firstline[256];
    infile.getline(firstline, 256);
    std::string sfirstline(firstline);
    std::cout << sfirstline;
    int procpos = sfirstline.find("PROC_NUM = ");
    int maxpos = sfirstline.find("MAX_MES_LEN = ");
    int steppos = sfirstline.find("MES_LEN_STEP = ");
    proc_num = atoi(sfirstline.substr(procpos + 11, maxpos - procpos - 13).c_str());
    max_mes_len = atoi(sfirstline.substr(maxpos + 14, steppos - maxpos - 16).c_str());
    mes_len_step = atoi(sfirstline.substr(steppos + 14, sfirstline.length() - steppos - 16).c_str());

}
