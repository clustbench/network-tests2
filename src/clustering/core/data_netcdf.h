#ifndef DATA_NETCDF_H
#define DATA_NETCDF_H

#include <cstdlib>
#include <netcdfcpp.h>
#include <string>

#include "data_abstract.h"

using namespace std;

class data_netcdf: public Data_Abstract
{
public:
    data_netcdf(string iFileName, string iHostsFileName = "");

    int getRealEndMessageLength();
    matrix getMatrix(int iMatrixLength);

private:
    NcFile *sourceFile;
};

#endif // DATA_NETCDF_H
