#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <set>
#include <string>

#include "clust.h"
#include "netcdf_file.h"
#include "arc_netcdf_file.h"

#define M 2
#define N 3
#define SIZE (N*N)

//using std::sort;
//using std::set;
//using std::set_intersection;
//using std::inserter;
using std::string;

#define random(p)	(rand ()%p)
#define PROG_NAME	"network_tests_clustering"

void print_format_string ()
{
	printf ("Use %s <NetCDF file> [flags]\n", PROG_NAME);
	printf ("-d <deviation> - floating point value, representing maximum of difference in vertical clusters. Default - 0.1\n");
	printf ("-p <precision> - floating point value. Maximum distance between elements in one cluster. Default - 0.001\n");
	printf ("-o <output filename> - set's output filename. Default - output.nc\n");
	printf ("-abs - absolute precision. Default - relative ((max - min_non_zero)*precision)\n");
	printf ("-fs <first union size> - number of matrices, used to create first partition. Default - 5\n");
	printf ("-os <other union size> - number of matrices, used to create another partitions. Default - 4\n");
	printf ("-arc - creates archivated output\n");
}

#define __ERROR_FILE_FORMAT printf ("Error file format\n"); print_format_string (); return 1;

int main (int argc, char **argv)
{
	clustering_params_t params;
	
    params.deviation = 0.1;
    params.first_union_size = 5;
    params.other_union_size = 4;
    params.precision = 0.001;
	params.flags = CLUST_PARAM_PREC_REL;
	bool is_archive = false;
	char *output_name = (char *)"output.nc";
	FileWriter *writer;
	
	if (argc < 2) {
		print_format_string ();
		return 0;
	} else {
		for (int i = 2; i < argc; ++i) {
			switch (argv [i][1]) {
				case 'a':
					switch (argv [i][2]) {
						case 'b':
							if (argv [i][3] == 's' && argv [i][4] == '\0') {
								params.flags = CLUST_PARAM_PREC_ABS; 
							} else {
								__ERROR_FILE_FORMAT
							}
							break;
						case 'r':
							if (argv [i][3] == 'c' && argv [i][4] == '\0') {
								is_archive = true;
							} else {
								__ERROR_FILE_FORMAT
							}
							break;
						default: __ERROR_FILE_FORMAT
					}
					/*if (argv [i][2] != 'b' || argv [i][3] != 's' || argv [i][4] != '\0') {
						__ERROR_FILE_FORMAT
					}*/
					
					break;
				case 'd':
					if (argv [i][2] != '\0' || i + 1 >= argc) {
						__ERROR_FILE_FORMAT
					}
					sscanf (argv [i+1], "%lg", &params.deviation);
					if (params.deviation < 0) {
						__ERROR_FILE_FORMAT
					}
					++i;
					break;
				case 'f':
					if (argv [i][2] != 's' || argv [i][3] != '\0' || i + 1 >= argc) {
						__ERROR_FILE_FORMAT
					}
					sscanf (argv [i+1], "%lu", &params.first_union_size);
					if (params.first_union_size == 0) {
						__ERROR_FILE_FORMAT
					}
					++i;
					break;
				case 'o':
					if (argv [i][2] == '\0' && i + 1 < argc) {
						output_name = argv [i+1];
					} else {
						if (argv [i][2] != 's' || argv [i][3] != '\0' || i + 1 >= argc) {
						__ERROR_FILE_FORMAT
						}
						sscanf (argv [i+1], "%lu", &params.other_union_size);
						if (params.other_union_size == 0) {
							__ERROR_FILE_FORMAT
						}
					}
					++i;
					break;
				case 'p':
					if (argv [i][2] != '\0' || i + 1 >= argc) {
						__ERROR_FILE_FORMAT
					}
					sscanf (argv [i+1], "%lg", &params.precision);
					if (params.precision <= 0) {
						__ERROR_FILE_FORMAT
					}
					++i;
					break;
				default: 
					__ERROR_FILE_FORMAT
			}
		}
	}
    
	try {
		FileReader reader (argv [1]);
		
		file_info_t file_info;
		reader.GetFileInfo(&file_info);
		
		if (is_archive) {
			writer = new ArcNetCDFFileWriter (output_name, argv [1]);
		} else {
			writer = new NetCDFFileWriter (output_name, argv [1]);;
			//writer = new StdFileWriter (file_info);
		}
		
		if (writer == NULL) {
			throw string ("Unexpected error - can't create writer object");
		}
		//StdFileWriter writer (file_info);
		
		RunClustering (reader, *writer, params);
		
		delete writer;
		
	} catch (string str) {
		printf ("ERROR:\n%s\n", str.data ());
		return 1;
	} catch (...) {
		printf ("Unknown Error\n");
	}
    
    return 0;
}
