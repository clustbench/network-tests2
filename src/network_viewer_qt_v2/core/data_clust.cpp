#include "data_clust.h"

#include <iostream>
ClusterReader::ClusterReader(std::string fname = "") : filename(fname)
{
}

std::vector < Cluster > ClusterReader::getClusters()
{
    return clusters;
}

void ClusterReader::readFromFile()
{
    std::ifstream infile(filename.c_str());
    char line[256] = {0};
    infile.getline(line, 256);
    std::string sfirstline(line);
    int procpos = sfirstline.find("PROC_NUM = ");
    int begpos = sfirstline.find("BEG_MES_LEN = ");
    int endpos = sfirstline.find("END_MES_LEN = ");
    int steppos = sfirstline.find("STEP_LEN = ");
    proc_num = atoi(sfirstline.substr(procpos + 11, begpos - procpos - 13).c_str());
    beg_mes_len = atoi(sfirstline.substr(begpos + 14, endpos - begpos - 16).c_str());
    end_mes_len = atoi(sfirstline.substr(endpos + 14, steppos - endpos - 16).c_str());
    step_len = atoi(sfirstline.substr(steppos + 11, sfirstline.length() - steppos - 11).c_str());
    std::vector < std::pair <int, int> > clustPairs;
    std::vector < double > clustM, clustD;
    while (!infile.eof()) {
        for (int i = 0; i < 256; i++)
            line[i] = 0;
        infile.getline(line, 256);
        sfirstline = std::string(line);
        int a = sfirstline.find("Cluster#");
        if (a == -1)
            return;
        char c = 0;
        int state = 0, first, second;
        std::string number;
        while (c != '\n') {
            infile.get(c);
            if (c == '(') {
                state = 1;
            } else if (c == ')') {
                state = 0;
                second = atoi (number.c_str());
                clustPairs.push_back(std::pair <int, int> (first, second));
                number.clear();
            } else if (c == ',' && state == 1) {
                state = 2;
                first = atoi (number.c_str());
                number.clear();
            } else if (isdigit(c)) {
                number.append(1, c);
            }
        }
        for (int i = 0; i < (end_mes_len - beg_mes_len)/step_len + 1; i++) {
            double m;
            infile >> m;
            clustM.push_back(m);
        }
        for (int i = 0; i < (end_mes_len - beg_mes_len)/step_len + 1; i++) {
            double d;
            infile >> d;
            clustD.push_back(d);
        }
        clusters.push_back(Cluster(clustPairs, clustM, clustD));
        clustPairs.clear(); clustM.clear(); clustD.clear();
        c = 0;
        while (c != '\n')
            infile.get(c);
    }
}

int ClusterReader::getProcNum()
{
    return proc_num;
}

int ClusterReader::getBegMesLen()
{
    return beg_mes_len;
}

int ClusterReader::getEndMesLen()
{
    return end_mes_len;
}

int ClusterReader::getStepLen()
{
    return step_len;
}


std::vector < std::pair <int, int> > Cluster::getData()
{
    return data;
}
