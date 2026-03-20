
#ifdef _GNU_SOURCE
#include <getopt.h>
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>

#include "clustbench_types.h"
#include "clustbench_config.h"
#include "clustbench_plugin_operations.h"
#include "parse_arguments.h"

#define MESSAGE_BEGIN_LENGTH 0
#define MESSAGE_END_LENGTH 300
#define NUM_REPEATS 10
#define MESSAGE_STEP 100
#define NO_MASH_TYPE 0
#define DEFAULT_SYNC_TYPE 0
#define DEFAULT_TIMER_TYPE 0

#define BENCHMARK_DIR CLUSTBENCH_BENCHMARKS_DIR "/interconnect"

static char default_file_name_prefix[512];


int print_benchmark_help_message (clustbench_benchmark_parameters_t *parameters) 
{
    clustbench_benchmark_pointers_t pointers;

    if(clustbench_open_benchmark(parameters->path_to_benchmark_code_dir, parameters->benchmark_name,&pointers))
    {
        fprintf(stderr,"print_individual_benchmark_parameters: can't get pointers for '%s'\n",
                parameters->benchmark_name);
        return 1;
    }
    
    int ret = pointers.print_help(parameters);
    
    clustbench_close_benchmark_lib(&pointers);
    
    return ret;
}
    

int print_network_test_help_message(clustbench_benchmark_parameters_t *parameters)
{
    if(parameters != NULL)
    {
        return print_benchmark_help_message(parameters);
    }
#ifdef _GNU_SOURCE
    printf("\nCommand line format for this program is:\n"
           "%s\n\t\t\t[{ -f | --file } <file> ]\n"
           "\t\t\t[{ -t | --type } <benchmark_name> ]\n"
           "\t\t\t[{ -b | --begin } <message_length> ]\n"
           "\t\t\t[{ -e | --end } <message_length> ]\n"
           "\t\t\t[{ -s | --step } <step> ]\n"
           "\t\t\t[{ -n | --num-iterations } <number of iterations> ]\n"
           "\t\t\t[{ -d | --dump } <list of types of statistics>]\n"
           "\t\t\t[{ -l | --list-benchmarks}]\n"
           "\t\t\t[{ -p | --path-to-benchmark} <path> ]\n"
           "\t\t\t[{ -h | --help } [<benchmark_name>]]\n"
           "\t\t\t[{ -v | --version }]\n","interconnect_benchmark");
#else

    printf("\nCommand line format for this program is:\n"
           "%s\n\t\t\t[ -f <file> ]\n"
           "\t\t\t[ -t <benchmark_name> ]\n"
           "\t\t\t[ -b <message_length> ]\n"
           "\t\t\t[ -e <message_length> ]\n"
           "\t\t\t[ -s <step> ]\n"
           "\t\t\t[ -n <number of iterations> ]\n"
           "\t\t\t[ -d <list of types of statistics> ]\n"
           "\t\t\t[ -l ] - print list of benchmarks\n"
           "\t\t\t[ -p  <path_to_benchmark> ]\n"
           "\t\t\t[ -h [<benchmark_name>] ]  - print help to program or help to benchmark\n"
           "\t\t\t[ -v ] - print version\n","interconnect_benchmark");
#endif
    /*printf("\n\nValues of parametrs:\n"
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
           "\t\t\t\td - save deviation\n"
           "\t\t\t\tz - dump all individual delays\n"
           "\t\t\tby default this string is 'lamd'\n");
    printf("\n"
           "help\t\t - this text or help text for benchmark\n"
           "version\t\t - types clustbench version\n\n\n"
           "clustbench version: %s\n\n\n",CLUSTBENCH_VERSION);*/

    return 0;
}

int parse_network_test_arguments(clustbench_benchmark_parameters_t *parameters,
                                 int argc,char **argv,int mpi_rank)
{
	int arg_val;
    struct tm *time_dump;
    size_t len;
    
    /*
     * Exit code for return from this function
     */
    int return_flag = 0;  

    /*
     * We set default file_name prefix to format delays-TZ-YYYY-MM-DD
     */
    time_t tmp_time = time(NULL);
    time_dump = localtime(&tmp_time);
    strcpy(default_file_name_prefix,"delays-");
    len = strlen("delays-");
    strftime(default_file_name_prefix+len,512-len,"%Z-%Y-%m-%d",time_dump);

    parameters->timer_type                 =  DEFAULT_TIMER_TYPE;
    parameters->sync_type                  =  DEFAULT_SYNC_TYPE;
    parameters->mash_type                  =  NO_MASH_TYPE;
	parameters->num_procs                  =  0; /* Special for program break on any error */
	parameters->benchmark_name             =  NULL;
    parameters->begin_message_length       =  MESSAGE_BEGIN_LENGTH;
    parameters->end_message_length         =  MESSAGE_END_LENGTH;
    parameters->step_length                =  MESSAGE_STEP;
    parameters->num_repeats                =  NUM_REPEATS;
    parameters->file_name_prefix           =  default_file_name_prefix;
    parameters->path_to_benchmark_code_dir =  NULL;
    parameters->benchmark_parameters       =  NULL;
    parameters->statistics_save            =  CLUSTBENCH_MIN     | 
                                              CLUSTBENCH_DEVIATION | 
                                              CLUSTBENCH_AVERAGE | 
                                              CLUSTBENCH_MEDIAN  |
                                              CLUSTBENCH_ALL;
    parameters->algorithm_main_info        =  NULL;

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

    int individual_options_flag = 0;
    opterr = 0;

    enum AlgorithmType algo_type = kEmptyAlgo;
    int frequency = -1;
    int windows_amount = -1;
    double ratio = -1;
    double target_criterion_value = -1;

    for ( ; ; )
    {
        if (individual_options_flag == 1) break;
#ifdef _GNU_SOURCE
        arg_val = getopt_long(argc,argv,"t:f:n:b:e:s:z:i:d:lp:h:v:a:o:w:c:r:m",options,NULL);
#else

        arg_val = getopt(argc,argv,"t:f:n:b:e:s:z:i:d:lp:h:v:a:o:w:c:r:m");
#endif

        if(arg_val == -1)
            break;

        switch(arg_val)
        {
            char *tmp_str;
        case 'b':
            parameters->begin_message_length = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str != '\0')
            {
                if(!mpi_rank)
                {
                    fprintf(stderr,"parse_parameters: Parse parameter with name 'begin' failed: '%s'",
                        strerror(errno));
                }
                return ERROR_FLAG;
            }
            break;
        case 'e':
            parameters->end_message_length = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str!='\0')
            {
                if(!mpi_rank)
                {
                    fprintf(stderr,"parse_parameters: Parse parameter with name 'end' failed: '%s'",
                        strerror(errno));
                }
                return ERROR_FLAG;
            }
            break;
        case 's':
            parameters->step_length = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str!='\0')
            {
                if(!mpi_rank)
                {
                    fprintf(stderr,"parse_parameters: Parse parameter with name 'step' failed: '%s'",
                        strerror(errno));
                }
                return ERROR_FLAG;
            }
            break;
        case 'z':
          if (strcmp(optarg, "default") == 0)
          {
            parameters->sync_type = 0;
          }
          else if (strcmp(optarg, "custom") == 0)
          {
            parameters->sync_type = 1;
          }
          else if (strcmp(optarg, "custom_offset") == 0)
          {
            parameters->sync_type = 2;
          }
          else{
            if(!mpi_rank)
            {
                fprintf(stderr, "parse_parameters: Parse parameter with name 'sync type' failed: '%s'",
                    strerror(errno));
            }
            return ERROR_FLAG;
          }            
          break;
        case 'i':
            parameters->timer_type = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str!='\0')
            {
                if(!mpi_rank)
                {
                    fprintf(stderr, "parse_parameters: Parse parameter with name 'num repeats' failed: '%s'",
                        strerror(errno));
                }
                return ERROR_FLAG;
            }            
            break;
        case 'm':
            parameters->mash_type = strtoul(optarg, &tmp_str, 10);
            printf("SET MASH TYPE: %d\n", parameters->mash_type);
            if(*tmp_str!='\0')
            {
                if(!mpi_rank)
                {
                    fprintf(stderr,"parse_parameters: Parse parameter with name 'mash' failed: '%s'",
                        strerror(errno));
                }
                return ERROR_FLAG;
            }
            break;
        case 'n':
            parameters->num_repeats = strtoul(optarg, &tmp_str, 10);
            if(*tmp_str!='\0')
            {
                if(!mpi_rank)
                {
                    fprintf(stderr, "parse_parameters: Parse parameter with name 'num repeats' failed: '%s'",
                        strerror(errno));
                }
                return ERROR_FLAG;
            }            
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
        case 'd':
            {
                parameters->statistics_save = 0;
                char *tmp;
                for(tmp = optarg; *tmp != '\0'; ++tmp)
                {
                    if(*tmp == 'l') /* min */
                    {
                        parameters->statistics_save |= CLUSTBENCH_MIN;
                        continue;
                    }
                    else if(*tmp == 'm') /* median */
                    {
                        parameters->statistics_save |= CLUSTBENCH_MEDIAN;
                        continue;
                    }
                    else if(*tmp == 'a') /* average */
                    {
                        parameters->statistics_save |= CLUSTBENCH_AVERAGE;
                        continue;
                    }
                    else if(*tmp == 'd') /* deviation */
                    {
                        parameters->statistics_save |= CLUSTBENCH_DEVIATION;
                        continue;
                    }
                    else if(*tmp == 'z') /* all individual delays */
                    {
                        parameters->statistics_save |= CLUSTBENCH_ALL;
                        continue;
                    }
                    
                    if(!mpi_rank)
                    {
                        fprintf(stderr,"for option statistics_save - unknown type of statistics '%c'\n",*tmp);
                    }
                    return ERROR_FLAG;
                }
            }
            break;
        case 't':
            parameters->benchmark_name = optarg;
            char *tmptmp = (char *)malloc((strlen(BENCHMARK_DIR) + strlen(parameters->benchmark_name) + 2) * sizeof(char));
            int i;
            for (i = 0; i < strlen(BENCHMARK_DIR); i++){
                tmptmp[i] = BENCHMARK_DIR[i];
            }
            tmptmp[i] = '/';
            i = i + 1;
            memmove(tmptmp+i, parameters->benchmark_name, strlen(parameters->benchmark_name) + 1);
            printf("%s\n", tmptmp);
            parameters->path_to_benchmark_code_dir = tmptmp;
		    break;
        case 'v':
            if(!mpi_rank)
            {
			    printf("Version: %s\n",CLUSTBENCH_VERSION);
            }
            return  VERSION_FLAG;
            break;
        case ':':
            if(!mpi_rank)
            {
                fprintf(stderr, "option -%c is missing a required argument.\n\n"
                                " Try to run program with -h option\n", optopt);
            }
            return ERROR_FLAG;
            break;
        case 'h':
            parameters->benchmark_name = optarg;
            char *mptmp = (char *)malloc((strlen(BENCHMARK_DIR) + strlen(parameters->benchmark_name) + 2) * sizeof(char));
            int j;
            for (j = 0; j < strlen(BENCHMARK_DIR); j++){
                mptmp[j] = BENCHMARK_DIR[j];
            }
            mptmp[j] = '/';
            j = j + 1;
            memmove(mptmp+j, parameters->benchmark_name, strlen(parameters->benchmark_name) + 1);
            printf("%s\n", mptmp);
            parameters->path_to_benchmark_code_dir = mptmp;
            return_flag = HELP_FLAG;
            break;
        case '?':
            if(optopt == 'h')
            {
                if(!mpi_rank)
                {
                    print_network_test_help_message(NULL);
                }
                return HELP_FLAG;
            } else {
                individual_options_flag = 1;
            }
            break;
        case 'a':
          printf("GOT OPTARG FOR ALGO: %s\n", optarg);
          if (strcmp(optarg, "ScalarMinAlgo") == 0)
          {
            algo_type = kScalarMinAlgo;
          }
          else if (strcmp(optarg, "ScalarAvgAlgo") == 0)
          {
            algo_type = kScalarAvgAlgo;
          }
          else if (strcmp(optarg, "ScalarDevAlgo") == 0)
          {
            algo_type = kScalarDevAlgo;
          }
          else if (strcmp(optarg, "ScalarMedAlgo") == 0)
          {
            algo_type = kScalarMedAlgo;
          }
          else if (strcmp(optarg, "SpectrumFFTAlgo") == 0)
          {
            algo_type = kSpectrumFFTAlgo;
          }
          else if (strcmp(optarg, "ModifiedSpectrumFFTAlgo") == 0)
          {
            algo_type = kModifiedSpectrumFFTAlgo;
          }
          else
          {
            if(!mpi_rank)
            {
                fprintf(stderr, "parse_parameters: Parse parameter with name 'algo type' failed: wrong algo type");
            }
            return ERROR_FLAG;
          }
          break;
        case 'o':
          printf("GOT OPTARG FOR FREQUENCY: %s\n", optarg);
          frequency = atoi(optarg);
          if (frequency == 0)
          {
            if(!mpi_rank)
            {
              fprintf(stderr, "parse_parameters: Parse parameter with name 'frequency' failed: '%s'",
              strerror(errno));
            }
            return ERROR_FLAG;
          }         
          break;
        case 'w':
          printf("GOT OPTARG FOR WINDOWS_AMOUNT: %s\n", optarg);
          windows_amount = atoi(optarg);
          if (windows_amount == 0)
          {
            if(!mpi_rank)
            {
              fprintf(stderr, "parse_parameters: Parse parameter with name 'windows_amount' failed: '%s'",
              strerror(errno));
            }
            return ERROR_FLAG;
          }        
          break;
        case 'r':
          printf("GOT OPTARG FOR RATIO: %s\n", optarg);
          ratio = strtod(optarg, &tmp_str);
          if ((ratio == 0) && (errno == ERANGE))
          {
            if(!mpi_rank)
            {
              fprintf(stderr, "parse_parameters: Parse parameter with name 'ratio' failed: '%s'",
              strerror(errno));
            }
            return ERROR_FLAG;
          }            
          break;
        case 'c':
          printf("GOT OPTARG FOR TARGET_CRITERION_VALUE: %s\n", optarg);
          target_criterion_value = strtod(optarg, &tmp_str);
          if ((target_criterion_value == 0) && (errno == ERANGE))
          {
            if(!mpi_rank)
            {
              fprintf(stderr, "parse_parameters: Parse parameter with name 'target_criterion_value' failed: '%s'",
              strerror(errno));
            }
            return ERROR_FLAG;
          }            
          break;
        }
    } /* end for */
    if (algo_type != kEmptyAlgo && ((frequency == -1) || (windows_amount == -1) || (ratio == -1) || (target_criterion_value == -1))) {
      if(!mpi_rank)
      {
        fprintf(stderr, "parse_parameters: to use algorithm for automative delay measurements amount determination -o, -w, -r and -c options shold be set");
      }
      return ERROR_FLAG;
    }
    else {
        printf("ALGO_PARAMETERS: %d, %d, %f, %f\n", frequency, windows_amount, ratio, target_criterion_value);
        parameters->algorithm_main_info = (struct AlgorithmMainInfo *)malloc(sizeof(struct AlgorithmMainInfo));
        parameters->algorithm_main_info = get_algorithm_main_info(algo_type, frequency, windows_amount, ratio, target_criterion_value);
    }
    //printf("%d\n", return_flag);
    if(return_flag == HELP_FLAG)
    {
        if(mpi_rank == 0 && print_network_test_help_message(parameters))
        {
            return ERROR_FLAG;
        }
        return return_flag;
    }

    int tmp_optind = optind;
    optind = 0;
    int res = parse_individual_benchmark_parameters(parameters, 
        (argc - tmp_optind == 0) ? 0 : argc - tmp_optind + 2, 
        argv + (tmp_optind - 2),
        mpi_rank);
    if (res != 0) 
    {
        if (res == UNKNOWN_FLAG)
        {
            printf("Unknown option found.\n");
            print_network_test_help_message(parameters);
        }
        return ERROR_FLAG;
    }

    return return_flag;
}

int print_network_test_parameters(clustbench_benchmark_parameters_t *parameters)
{
    /*if (parameters == NULL) 
    {
        printf("Parameters not specified\n");
        return 1;
    }*/
    
    printf("Benchmark common parameters:\n");
    printf("\tbenchmark name = \"%s\"\n", parameters->benchmark_name);
    printf("\tpath to benchmark directory = \"%s\"\n\n",parameters->path_to_benchmark_code_dir);
    printf("\tnumber proceses = %d\n", parameters->num_procs);
    printf("\tbegin message length = \t\t%d\n", parameters->begin_message_length);
    printf("\tend message length = \t\t%d\n", parameters->end_message_length);
    printf("\tstep length = \t\t\t%d\n", parameters->step_length);
    printf("\tnumber of repeates = \t\t%d\n\n",parameters->num_repeats);

    if(parameters->statistics_save & CLUSTBENCH_MIN)
    {
        printf("\tresult file minimum = \t\t\"%s_min.nc\"\n",parameters->file_name_prefix);
    }

    if(parameters->statistics_save & CLUSTBENCH_AVERAGE)
    {
        printf("\tresult file average = \t\t\"%s_average.nc\"\n",parameters->file_name_prefix);
    }

    if(parameters->statistics_save & CLUSTBENCH_MEDIAN)
    {
        printf("\tresult file median = \t\t\"%s_median.nc\"\n",parameters->file_name_prefix);
    }
    
    if(parameters->statistics_save & CLUSTBENCH_DEVIATION)
    {
        printf("\tresult file deviation = \t\"%s_deviation.nc\"\n",parameters->file_name_prefix);
    }
    printf("\tresult file hosts\t\t\"%s_hosts.txt\"\n\n",parameters->file_name_prefix);

    printf("Individual benchmark parameters:\n");
    return print_individual_benchmark_parameters(parameters);

    return 0;
}

int print_individual_benchmark_parameters(clustbench_benchmark_parameters_t *parameters)
{
    
    int ret=0;

    clustbench_benchmark_pointers_t pointers;

    if(clustbench_open_benchmark(parameters->path_to_benchmark_code_dir, parameters->benchmark_name,&pointers))
    {
        fprintf(stderr,"print_individual_benchmark_parameters: can't get pointers for '%s'\n",
                parameters->benchmark_name);
        return 1;
    }

    ret = pointers.print_parameters(parameters);

    clustbench_close_benchmark_lib(&pointers); 
   
    return ret;
}

int parse_individual_benchmark_parameters(clustbench_benchmark_parameters_t *parameters,
                                          int argc, char **argv,int mpi_rank)
{
    
    int ret = 0;

    clustbench_benchmark_pointers_t pointers;

    if(parameters->benchmark_name==NULL)
    {
        if(!mpi_rank)
        {
            fprintf(stderr,"Name of benchmark is not specified.\n");
        }
        return 1;
    }

    if((argc == 0) || (argv == NULL) )
    {
        if(!mpi_rank)
        {
            printf("No individual parameters for benchmark '%s'\n",parameters->benchmark_name);
        }
    }

    if(clustbench_open_benchmark(parameters->path_to_benchmark_code_dir, parameters->benchmark_name,&pointers))
    {
        fprintf(stderr,"print_individual_benchmark_parameters: can't get pointers for '%s'\n",
                parameters->benchmark_name);
        return 1;
    }

    ret = pointers.parse_parameters(parameters,argc,argv,mpi_rank);
    clustbench_close_benchmark_lib(&pointers);         
   
    return ret;
}

