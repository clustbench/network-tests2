#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "benchmarks_common.h"
#include "clustbench_types.h"


int clustbench_time_cmp(const void *a, const void *b)
{
    clustbench_time_t val_a=*(clustbench_time_t *)a;
    clustbench_time_t val_b=*(clustbench_time_t *)b;

    if((val_a - val_b)>0) return 1;
    else if((val_a - val_b)<0) return -1;
    else return 0;
}

int clustbench_create_hosts_file
(
    const clustbench_benchmark_parameters_t *parameters,
    char **hosts_names
)
{
    FILE *f;
    char *file_name;
    int i;

    file_name=(char *)malloc((strlen(parameters->file_name_prefix)+strlen("_hosts.txt")+1)*sizeof(char));
    if(file_name==NULL)
    {
    	fprintf(stderr, "Memory allocation error\n");
    	return -1;
    }

    sprintf(file_name,"%s_hosts.txt",parameters->file_name_prefix);
    
    f=fopen(file_name,"w");
    if(f==NULL)
    {
    	fprintf(stderr,"create_test_hosts_file: File '%s' open error: %s\n\n", file_name, strerror(errno));
    	return -1;
    }

    for(i=0;i<parameters->num_procs;i++)
    {
    	fprintf(f,"%s\n",hosts_names[i]);
    }

    fclose(f);
    free(file_name);

    return 0;
}

