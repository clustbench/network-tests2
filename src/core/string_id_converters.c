#include <stdlib.h>
#include <string.h>

#include "string_id_converters.h"

static const char* file_data_types[NUM_NETWORK_TEST_DATATYPES+1]=
{
    "unknown_datatype",
    "average",
    "median",
    "deviation",
    "min",
    "all"
};


const char* file_data_type_to_string(const int data_type)
{
    if((data_type>0)&&(data_type<=NUM_NETWORK_TEST_DATATYPES))
        return file_data_types[data_type];
    else
        return file_data_types[0];
}
