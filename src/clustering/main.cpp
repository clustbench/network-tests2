#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <set>
#include <string>

#include <getopt.h>

#include "clust.h"
#include "netcdf_file.h"
#include "arc_netcdf_file.h"

#define M 2
#define N 3
#define SIZE (N*N)

using std::string;

#define random(p)   (rand ()%p)
#define PROG_NAME   "network_tests_clustering"

void print_format_string ()
{
	printf ("Use %s <NetCDF file> [flags]\n", PROG_NAME);
	printf ("Global arguments:\n%s\nParameters:\n%s%s%s%s%s%s%s",
		"\t-d, --deviation=NUMBER  \tfloating point value, representing maximum of difference in vertical clusters. Default - 0.1\n",
		"\t-p, --precision=NUMBER  \tfloating point value. Maximum distance between elements in one cluster. Default - 0.001\n",
		"\t-o, --output=FILENAME   \tset's output filename. Default - output.nc\n",
		"\t-a, --abs               \tabsolute precision. Default - relative ((max - min_non_zero)*precision)\n",
		"\t-f, --first-size=NUMBER \tnumber of matrices, used to create first partition. Default - 5\n",
		"\t-s, --other-size=NUMBER \tnumber of matrices, used to create another partitions. Default - 4\n",
		"\t-c, --compress          \tcreates archivated output\n"
	);
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

	const char* short_options = "d:p:o:af:s:ch";
	const struct option long_options[] = {
		{ "deviation",  required_argument, NULL, 'd' },
		{ "precision",  required_argument, NULL, 'p' },
		{ "output",     required_argument, NULL, 'o' },
		{ "absolute",   no_argument,       NULL, 'a' },
		{ "first-size", required_argument, NULL, 'f' },
		{ "other-size", required_argument, NULL, 's' },
		{ "compress",   no_argument,       NULL, 'c' },
		{ "help",       no_argument,       NULL, 'h' },
	};

	int current, option_index;
	while ((current = getopt_long(argc, argv, short_options,
		long_options, &option_index)) != -1)
	{
		switch(current) {
			case 'd':
				sscanf(optarg, "%lg", &params.deviation);
				if (params.deviation < 0) {
					__ERROR_FILE_FORMAT
				}
				break;
			case 'p':
				sscanf(optarg, "%lg", &params.precision);
				if (params.precision <= 0) {
					__ERROR_FILE_FORMAT
				}
				break;
			case 'o':
				output_name = optarg;
				break;
			case 'a':
				params.flags = CLUST_PARAM_PREC_ABS;
				break;
			case 'f':
				sscanf(optarg, "%lu", &params.first_union_size);
				if (params.first_union_size == 0) {
					__ERROR_FILE_FORMAT
				}
				break;
			case 's':
				sscanf(optarg, "%lu", &params.other_union_size);
				if (params.other_union_size == 0) {
					__ERROR_FILE_FORMAT
				}
				break;
			case 'c':
				is_archive = true;
				break;
			case 'h':
				print_format_string();
				break;
			default:
				__ERROR_FILE_FORMAT
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
