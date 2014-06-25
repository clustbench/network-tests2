#include <stdlib.h>
#include <stdio.h>
#include "data_abstract.h"
#include "data_netcdf.h"

data_netcdf::data_netcdf(string iFileName, string iHostsFileName)
{
    source_file_name = iFileName;
    sourceFile = new NcFile(source_file_name.c_str());

    if (!sourceFile->is_valid())
        throw "file \"" + iFileName + "\" is not valid NetCDF file";

#define FORMATCHECK(x, y, z) try{ x = sourceFile->get_var( y )->as_int(0);} catch(NcError) { throw "file \"" + iFileName + "\" have incorrect format: " + z ; return;}

//wtf???
//    if (sourceFile->num_dims() != 3  || sourceFile->num_vars() != 11)
//         throw "file \"" + iFileName + "\" have incorrect format: wrong count of variables and/or dimensions";

    test_type = sourceFile->get_var("test_type")->as_string(0);

    data_type = sourceFile->get_var("data_type")->as_string(0);

    FORMATCHECK(num_processors, "proc_num", "cannot get count of processors");
    FORMATCHECK(begin_message_length, "begin_mes_length", "cannot get begin message length");
    FORMATCHECK(end_message_length, "end_mes_length", "cannot get end message length");
    FORMATCHECK(step_length, "step_length", "cannot get step length");
    FORMATCHECK(noise_message_length, "noise_mes_length", "cannot get noise message length");
    FORMATCHECK(noise_message_num, "num_noise_mes", "cannot get count of noise messages");
    FORMATCHECK(noise_processors, "num_noise_proc", "cannot get count of noise processors");
    FORMATCHECK(num_repeats, "num_repeates", "cannot get count of repeats");

    if (iHostsFileName != "") {
        FILE *tmpHostFile;
        if ((tmpHostFile = fopen(iHostsFileName.c_str(), "r")) != NULL) {
            char tmpStr[256];
            while (!feof(tmpHostFile))
                if (fgets(tmpStr, 256, tmpHostFile) != NULL)
                    host_names.insert(host_names.end(), tmpStr);
            fclose(tmpHostFile);
        } else {
            throw "file with hosts \"" + iHostsFileName + "\" cannot be open and read. Host names are not avilable.";
        }
    }
}

int data_netcdf::getRealEndMessageLength()
{
    return (int)sourceFile->get_dim("n")->size() * step_length + begin_message_length;
}

Data_Abstract::matrix data_netcdf::getMatrix(int iMatrixLength)
{
    iMatrixLength /= step_length;
    Data_Abstract::matrix res;
    double mas[num_processors][num_processors];

    NcVar *t = sourceFile->get_var("data");
    t->set_cur(iMatrixLength);
    t->get(&mas[0][0], 1, num_processors, num_processors);

    for (int x = 0; x != num_processors; ++x) {
        list<double> tmpRow;
        for (int y = 0; y != num_processors; ++y)
            tmpRow.push_back(mas[x][y]);
        res.push_back(tmpRow);
    }

    return res;
}
