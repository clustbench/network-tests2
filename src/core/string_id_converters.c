#include <stdlib.h>

#include "string_id_converters.h"

static const char* file_data_types[NUM_NETWORK_TEST_DATATYPES+1]=
{
    "unknown_datatype",
    "average",
    "median",
    "deviation",
    "min"
};


const char* file_data_type_to_sring(const int data_type)
{
    if((data_type>0)&&(data_type<=NUM_NETWORK_TEST_DATATYPES))
        return file_data_types[data_type];
    else
        return file_data_types[0];
}


int get_test_type(const char *str)
{
    if(str==NULL) return -1;
    if(!strcmp(str,"one_to_one"))
        return ONE_TO_ONE_TEST_TYPE;
    if(!strcmp(str,"all_to_all"))
        return ALL_TO_ALL_TEST_TYPE;
    if(!strcmp(str,"async_one_to_one"))
        return ASYNC_ONE_TO_ONE_TEST_TYPE;
    if(!strcmp(str,"send_recv_and_recv_send"))
        return SEND_RECV_AND_RECV_SEND_TEST_TYPE;
    if(!strcmp(str,"noise"))
        return NOISE_TEST_TYPE;
    if(!strcmp(str,"noise_blocking"))
        return NOISE_BLOCKING_TEST_TYPE;
    if(!strcmp(str,"bcast"))
        return BCAST_TEST_TYPE;
    if(!strcmp(str,"put"))
        return PUT_ONE_TO_ONE_TEST_TYPE;
    if(!strcmp(str,"get"))
        return GET_ONE_TO_ONE_TEST_TYPE;
    return UNKNOWN_TEST_TYPE;
}

int get_test_type_name(int test_type,char *str)
{
    if(str==NULL) return -1;
    switch(test_type)
    {
    case ONE_TO_ONE_TEST_TYPE:
        strcpy(str,"one_to_one");
        break;
    case ASYNC_ONE_TO_ONE_TEST_TYPE:
        strcpy(str,"async_one_to_one");
        break;
    case ALL_TO_ALL_TEST_TYPE:
        strcpy(str,"all_to_all");
        break;
    case SEND_RECV_AND_RECV_SEND_TEST_TYPE:
        strcpy(str,"send_recv_end_recv_send");
        break;
    case NOISE_TEST_TYPE:
        strcpy(str,"noise");
        break;
    case NOISE_BLOCKING_TEST_TYPE:
        strcpy(str,"noise_blocking");
        break;
    case BCAST_TEST_TYPE:
        strcpy(str,"bcast");
        break;
    case GET_ONE_TO_ONE_TEST_TYPE:
        strcpy(str,"get");
        break;
    case PUT_ONE_TO_ONE_TEST_TYPE:
        strcpy(str,"put");
        break;
    default:
        strcpy(str,"unknown");
        break;
    }
    return 0;
}
