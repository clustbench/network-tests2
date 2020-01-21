
#ifdef _GNU_SOURCE
#include <getopt.h>
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "libconfig.h"

#include "types.h"
#include "../../parus_config.h"
#include "string_id_converters.h"

#define MESSAGE_BEGIN_LENGTH 0
#define MESSAGE_END_LENGTH 10000
#define NUM_REPEATS 100
#define MESSAGE_STEP 100
#define NOISE_MESSAGE_LENGTH 0
#define NOISE_MESSAGE_NUM 1
#define NUM_NOISE_PROCS 0

#define INFO_FLAG   5
#define VERSION_FLAG 1
#define ERROR_FLAG   -1
#define HELP_FLAG    2

const char *default_file_name_prefix = "network";


int print_network_test_help_message(void)
{
#ifdef _GNU_SOURCE
    printf("\nCommand line format for this program is:\n"
           "%s\n\t\t\t[{ -f | --file } <file> ]\n"
           "\t\t\t[{ -t | --type } <test_type> ]\n"
           "\t\t\t[{ -b | --begin } <message_length> ]\n"
           "\t\t\t[{ -e | --end } <message_length> ]\n"
           "\t\t\t[{ -s | --step } <step> ]\n"
           "\t\t\t[{ -l | --length_noise_message } <length> ]\n"
           "\t\t\t[{ -m | --num_noise_message } <number of noise messages> ]\n"
           "\t\t\t[{ -p | --procs_noise } <number of noise MPI processes> ]\n"
           "\t\t\t[{ -n | --num_iterations } <number of iterations> ]\n"
           "\t\t\t[{ -h | --help }]\n"
           "\t\t\t[{ --info      }]\n"
           "\t\t\t[{ --fullinfo  }]\n"
           "\t\t\t[{ -v | --version }]\n","network_test2");


#else

    printf("\nCommand line format for this program is:\n"
           "%s\n\t\t\t[ -f <file> ]\n"
           "\t\t\t[ -t <test_type> ]\n"
           "\t\t\t[ -b <message_length> ]\n"
           "\t\t\t[ -e <message_length> ]\n"
           "\t\t\t[ -s <step> ]\n"
           "\t\t\t[ -l <noise message length> ]\n"
           "\t\t\t[ -m <number of noise message> ]\n"
           "\t\t\t[ -p <number of noise processes> ]\n"
           "\t\t\t[ -n <number of iterations> ]\n"
           "\t\t\t[ -h ] - print help\n"
           "\t\t\t[ --info] - print short information\n"
           "\t\t\t[ --fullinfo] - print full information\n"
           "\t\t\t[ -v ] - print version\n","network_test2");
#endif
    printf("\n\nValues of parametrs:\n"
           "file\t\t - default  prefix for files with test results is %s/network\n", PARUS_DATA_DIR);
    printf("type\t\t - default one_to_one. This parametr sets type of test that will\n"
           "\t\t\tbe run on multiprocessor system.\n"
           "\t\t\tYou able to show one of some words as value of parametr:\n"
           "\t\t\tone_to_one - is a test that lock process when translate data\n"
           "\t\t\t\tfrom one processor to other.\n"
           "\t\t\tasync_one_to_one - is a test that not lock process when\n"
           "\t\t\t\ttranslate data from one processor to other.\n"
           "\t\t\t\tData transfer process is called simultaniously\n"
           "\t\t\t\t on two processors against to other.\n"
           "\t\t\tsend_recv_and_recv_send - is a test that lock process when\n"
           "\t\t\t\ttranslate data from one processor to other\n"
           "\t\t\t\tand back.\n"
           "\t\t\t\tAs result we measure time between sending from one processor\n"
           "\t\t\t\tto other and came back this message from peer processor.\n"
           "\t\t\tall_to_all - is a test that translate data simulteniously to\n"
           "\t\t\t\tall other processes.\n"
           "\t\t\tnoise - is a test where some processors generate noise.\n"
           "\t\t\t\tThis test works like async_one_to_one test.\n"
           "\t\t\tnoise_blocking - is a test where some processors generate noise.\n"
           "\t\t\t\tThis test works like one_to_one test.\n"
           "\t\t\tbcast - is a test with broadcast messages.\n"
	   "\t\t\tput - MPI_Put function testing.\n"
	   "\t\t\tget - MPI_Get function testing.\n"
	   "\t\t\t"
           "\n"
           "begin\t\t\t - sets begin message length, '%d' by default\n", (int)MESSAGE_BEGIN_LENGTH);
    printf("end\t\t\t - sets end message length, '%d' by default\n", (int)MESSAGE_END_LENGTH);
    printf("step\t\t\t - sets step in grow message length process,'%d' by default\n",(int)MESSAGE_STEP);
    printf("length_noise_message\t - sets a length of noise message in noise tests.\n"
           "\t\t\t\tIf test type is not one of noise or noise_blocking this argument\n"
           "\t\t\t\twill be ignored. The default value is '%d'.\n", (int)NOISE_MESSAGE_LENGTH);
    printf("num_noise_message\t - sets the number of noise messages in each interaction iteration in noise tests.\n"
           "\t\t\t\tIf test type is not one of noise or noise_blocking this argument\n"
           "\t\t\t\twill be ignored. The default value is '%d'.\n", (int)NOISE_MESSAGE_NUM);
    printf("procs_noise\t\t - number of noise processors in noise tests.\n"
           "\t\t\t\tIf test type is not one of noise or noise_blocking this argument\n"
           "\t\t\t\twill be ignored. The default value is '%d'.\n", (int)NUM_NOISE_PROCS);

    printf("num_repeats\t\t - sets number iteration in send process, '%d' by default\n",(int)NUM_REPEATS);
    printf("\n"
           "info\t\t - print short information about test\n"
           "fullinfo\t - print full information about test\n"
           "help\t\t - this text\n"
           "version\t\t - types parus version\n\n\n"
           "Parus version: %s\n\n\n",PARUS_VERSION);

    return 0;
}

int print_info_test(struct network_test_parameters_struct *parameters ,int d)
{
    char test[30];
    char library[100] = "\0";
    char tests[10] = "tests/";
    char libso[5] = ".so";
    char liblib[5] ="/lib";
    char info[12] = "/config.cfg";
    config_t cfg;
    const char *info_str;
    const char *info_parameters;
    if (get_test_type_name(parameters->test_type, test) == -1)
        return -1;
    strcat(library, tests);
    strcat(library, test);
    library[strlen(library)] = '//';
    strcat(library, info);
    
    config_init(&cfg);
    if (! config_read_file(&cfg, library))
    {
        return -1;
    }
    if (d==0)
    {
        if (config_lookup_string(&cfg, "info", &info_str))
            printf("%s\n", info_str);
        if (config_lookup_string(&cfg, "parameters", &info_parameters))
            printf("%s\n", info_parameters);
    }
    else 
    {
        if (config_lookup_string(&cfg, "full_info", &info_str))
            printf("%s\n", info_str);
        if (config_lookup_string(&cfg, "full_parameters", &info_parameters))
            printf("%s\n", info_parameters);
    }
    
    config_destroy(&cfg);
    return 0;
}

int parse_network_test_arguments(int argc,char **argv,struct network_test_parameters_struct *parameters)
{
	int arg_val;

	parameters->num_procs            =  0; /* Special for program break on any error */
	parameters->test_type            =  ONE_TO_ONE_TEST_TYPE;
    parameters->begin_message_length =  MESSAGE_BEGIN_LENGTH;
    parameters->end_message_length   =  MESSAGE_END_LENGTH;
    parameters->step_length          =  MESSAGE_STEP;
    parameters->num_repeats          =  NUM_REPEATS;
    
    parameters->noise_message_length  =  NOISE_MESSAGE_LENGTH;
    parameters->num_noise_messages   =  NOISE_MESSAGE_NUM;
    parameters->num_noise_procs      =  NUM_NOISE_PROCS;
    parameters->file_name_prefix     =  default_file_name_prefix;

#ifdef _GNU_SOURCE

    struct option options[15]=
    {
        {"type",required_argument,NULL,'t'},
        {"file",required_argument,NULL,'f'},
        {"num_iterations",required_argument,NULL,'n'},
        {"begin",required_argument,NULL,'b'},
        {"end",required_argument,NULL,'e'},
        {"step",required_argument,NULL,'s'},
        {"length_noise_message",required_argument,NULL,'l'},
        {"num_noise_message",required_argument,NULL,'m'},
        {"procs_noise",required_argument,NULL,'p'},
        {"version",no_argument,NULL,'v'},
        {"help",no_argument,NULL,'h'},
        {"resume",no_argument,NULL,'r'},
        {"ignore",no_argument,NULL,'i'},
        {0,0,0,0}
    };
#endif

    int j;
    int k =0;
    int test_param_type = ONE_TO_ONE_TEST_TYPE;
    for (j=1;j<argc;j++)
    {

        if (strcmp(argv[j], "-t")==0)
        {
            if ((j+1) != argc)
            {
                if ((test_param_type = get_test_type(argv[j+1])) == UNKNOWN_TEST_TYPE)
                    test_param_type = ONE_TO_ONE_TEST_TYPE;
                k=1;
                break;
            }
        }
    }
    
    char params[40];

#ifdef _GNU_SOURCE
    if ((test_param_type  == NOISE_TEST_TYPE) || (test_param_type == NOISE_BLOCKING_TEST_TYPE))
        strcat(params,"t:f:n:b:e:s:l:m:p:h:v:r:-");
    else
        strcat(params,"t:f:n:b:e:s:h:v:r:-");
#else
    if ((test_param_type  == NOISE_TEST_TYPE) || (test_param_type == NOISE_BLOCKING_TEST_TYPE))
        strcat(params,"t:f:n:b:e:s:l:m:p:h:v:r:-");
    else
        strcat(params,"t:f:n:b:e:s:h:v:r:-");
#endif

    
    for ( ; ; )
    {
#ifdef _GNU_SOURCE
        arg_val = getopt_long(argc,argv,params,options,NULL);
#else

        arg_val = getopt(argc,argv,params);
#endif
        
    int i;
        if ( arg_val== -1 )
            break;

        switch ( arg_val )
        {
        case 'b':
            parameters->begin_message_length = atoi(optarg);
            break;
        case 'e':
            parameters->end_message_length = atoi(optarg);
            break;
        case 's':
            parameters->step_length = atoi(optarg);
            break;
        case 'l':
            parameters->noise_message_length = atoi(optarg);
            break;
        case 'm':
            parameters->num_noise_messages = atoi(optarg);
            break;
        case 'p':
            parameters->num_noise_procs = atoi(optarg);
            break;
        case 'n':
            parameters->num_repeats = atoi(optarg);
            break;
        case 'f':
            parameters->file_name_prefix = optarg;
            break;
        case 't':
            if ( ( parameters->test_type = get_test_type(optarg) ) == UNKNOWN_TEST_TYPE )
                parameters->test_type = ONE_TO_ONE_TEST_TYPE;
			break;
        case 'v':
			printf("Version: %s\n",PARUS_VERSION);
            return  VERSION_FLAG;
            break;
        case 'h':
            print_network_test_help_message();
            return  HELP_FLAG;
            break;
        case '?':
            print_network_test_help_message();
            return ERROR_FLAG;
           break;
        case '-':
            for(i=1;i<argc;i++)
            {
                if (strcmp(argv[i], "--info") == 0)
                {       
                        if (print_info_test(parameters, 0) == -1)
                            return ERROR_FLAG;
                        else
                            return INFO_FLAG;
                        continue;
                }
                if (strcmp(argv[i], "--fullinfo")== 0)
                {       
                        if (print_info_test(parameters, 1) == -1)
                            return ERROR_FLAG;
                        else
                            return INFO_FLAG;
                        continue;
                }
                
            }
        }

    }

    return 0;
}


