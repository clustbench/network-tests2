
#ifdef _GNU_SOURCE
#include <getopt.h>
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include "clustbench_types.h"
#include "clustbench_config.h"

#define MESSAGE_BEGIN_LENGTH 0
#define MESSAGE_END_LENGTH 10000
#define NUM_REPEATS 100
#define MESSAGE_STEP 100

#define BENCHMARK_DIR CLUSTBENCH_BENCHMARKS_DIR "/interconnect"

#define VERSION_FLAG 1
#define ERROR_FLAG   -1
#define HELP_FLAG    2
#define LIST_FLAG    3

const char *default_file_name_prefix = "interconnect";


int print_network_test_help_message(const char *benchmark_name)
{
    if(benchmark_name != NULL)
    {
        return print_benchmark_help_message(benchmark_name);
    }
#ifdef _GNU_SOURCE
    printf("\nCommand line format for this program is:\n"
           "%s\n\t\t\t[{ -f | --file } <file> ]\n"
           "\t\t\t[{ -t | --type } <benchmark_name> ]\n"
           "\t\t\t[{ -b | --begin } <message_length> ]\n"
           "\t\t\t[{ -e | --end } <message_length> ]\n"
           "\t\t\t[{ -s | --step } <step> ]\n"
           "\t\t\t[{ -n | --num-iterations } <number of iterations> ]\n"
           "\t\t\t[{ -d | --dump } <list of types of statistics>]"
           "\t\t\t[{ -l | --list-benchmarks}]\n"
           "\t\t\t[{ -p | --path-to-benchmark} <path> ]\n"
           "\t\t\t[{ -h | --help } [<benchmark_name>]]\n"
           "\t\t\t[{ -v | --version }]\n","network_test2");
#else

    printf("\nCommand line format for this program is:\n"
           "%s\n\t\t\t[ -f <file> ]\n"
           "\t\t\t[ -t <benchmark_name> ]\n"
           "\t\t\t[ -b <message_length> ]\n"
           "\t\t\t[ -e <message_length> ]\n"
           "\t\t\t[ -s <step> ]\n"
           "\t\t\t[ -n <number of iterations> ]\n"
           "\t\t\t[ -d <list of types of statistics> ]"
           "\t\t\t[ -l ] - print list of benchmarks\n"
           "\t\t\t[ -p  <path_to_benchmark> ]\n"
           "\t\t\t[ -h [<benchmark_name>] ]  - print help to program or help to benchmark\n"
           "\t\t\t[ -v ] - print version\n","network_test2");
#endif
    printf("\n\nValues of parametrs:\n"
           "file\t\t - default  prefix for files with test results is %s/network\n", CLUSTBENCH_DATA_DIR);
    printf("type\t\t - by this parametr user sets benchmark that will be run\n"
           "\t\t\ton multiprocessor system.\n"
           "\t\t\tThe list of availavble benchmarks possible to see with -l or --list-benchmarks\n"
           "\t\t\tparameter\n"
           "\n"
           "begin\t\t\t - sets begin message length, '%d' by default\n", (int)MESSAGE_BEGIN_LENGTH);
    printf("end\t\t\t - sets end message length, '%d' by default\n", (int)MESSAGE_END_LENGTH);
    printf("step\t\t\t - sets step in grow message length process,'%d' by default\n",(int)MESSAGE_STEP);
    printf("num_repeats\t\t - sets number iteration in send process, '%d' by default\n",(int)NUM_REPEATS);
    printf("dump\t\t\t - sets list of statistics to save into the files. It is a string with symbols:\n"
           "\t\t\t\tl - save min delay values\n"
           "\t\t\t\ta - save average\n"
           "\t\t\t\tm - save median\n"
           "\t\t\t\td - save dispersion\n"
           "\t\t\t\tz - dump all individual delays\n"
           "\t\t\tby default this string is 'lamd'\n");
    printf("\n"
           "help\t\t - this text or help text for benchmark\n"
           "version\t\t - types clustbench version\n\n\n"
           "clustbench version: %s\n\n\n",CLUSTBENCH_VERSION);

    return 0;
}

int parse_network_test_arguments(int argc,char **argv,clustbench_benchmark_parameters_t *parameters)
{
	int arg_val;

	parameters->num_procs                  =  0; /* Special for program break on any error */
	parameters->benchmark_name             =  NULL;
    parameters->begin_message_length       =  MESSAGE_BEGIN_LENGTH;
    parameters->end_message_length         =  MESSAGE_END_LENGTH;
    parameters->step_length                =  MESSAGE_STEP;
    parameters->num_repeats                =  NUM_REPEATS;
    parameters->file_name_prefix           =  default_file_name_prefix;
    parameters->path_to_benchmark_code_dir =  BENCHMARK_DIR;
    parameters->benchmark_parameters       =  NULL;
    parameters->statistics_save            =  CLUSTBENCH_MIN     | CLUSTBENCH_DISPERSION | 
                                              CLUSTBENCH_AVERAGE | CLUSTBENCH_MEDIAN;

#ifdef _GNU_SOURCE

    struct option options[]=
    {
        {"type",required_argument,NULL,'t'},
        {"file",required_argument,NULL,'f'},
        {"num_iterations",required_argument,NULL,'n'},
        {"begin",required_argument,NULL,'b'},
        {"end",required_argument,NULL,'e'},
        {"step",required_argument,NULL,'s'},
        {"dump",required_argument,NULL,'d'},
        {"list-benchmarks",no_argument,NULL,'l'},
        {"path-to-benchmark",required_argument,NULL,'p'},
        {"version",no_argument,NULL,'v'},
        {"help",optional_argument,NULL,'h'},
        {0,0,0,0}
    };
#endif

    for ( ; ; )
    {
#ifdef _GNU_SOURCE
        arg_val = getopt_long(argc,argv,"t:f:n:b:e:s:l:p:h:v",options,NULL);
#else

        arg_val = getopt(argc,argv,"t:f:n:b:e:s:l:p:h:v");
#endif

        if ( arg_val== -1 )
            break;

        switch ( arg_val )
        {
            char *tmp_str;
        case 'b':
            parameters->begin_message_length = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str!='\0')
            {
                fprintf(stderr,"parse_parameters: Parse parameter with name 'begin' failed: '%s'",
                        strerror(errno));
                return ERROR_FLAG;
            }
            break;
        case 'e':
            parameters->end_message_length = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str!='\0')
            {
                fprintf(stderr,"parse_parameters: Parse parameter with name 'end' failed: '%s'",
                        strerror(errno));
                return ERROR_FLAG;
            }
            break;
        case 's':
            parameters->step_length = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str!='\0')
            {
                fprintf(stderr,"parse_parameters: Parse parameter with name 'step' failed: '%s'",
                        strerror(errno));
                return ERROR_FLAG;
            }
            break;
        case 'n':
            parameters->num_repeats = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str!='\0')
            {
                fprintf(stderr,"parse_parameters: Parse parameter with name 'num repeats' failed: '%s'",
                        strerror(errno));
                return ERROR_FLAG;
            }
            break;
           break;
        case 'l':
            return LIST_FLAG;
            break;
        case 'p':
            parameters->path_to_benchmark_code_dir = optarg;
            break;
        case 'f':
            parameters->file_name_prefix = optarg;
            break;
        case 't':
            parameters->benchmark_name = optarg;
			break;
        case 'v':
			printf("Version: %s\n",PARUS_VERSION);
            return  VERSION_FLAG;
            break;
        case ':':
            if(optopt == 'h')
            {
                print_network_test_help_message(NULL);
                return HELP_FLAG;
            }
            
            fprintf(stderr, "option -%c is missing a required argument.\n\n"
                            " Try to run program with -h option\n", optopt);
            return ERROR_FLAG;
            break;
        case 'h':
            if(print_network_test_help_message(optarg))
            {
                return ERROR_FLAG;
            }
            return  HELP_FLAG;
            break;
        case '?':
            print_network_test_help_message();
            return ERROR_FLAG;
            break;
        }

    } /* end for */

    return 0;
}


