#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "my_time.h"
#include "types.h"


int my_time_cmp(const void *a, const void *b)
{
    px_my_time_type val_a=*(px_my_time_type *)a;
    px_my_time_type val_b=*(px_my_time_type *)b;

    if((val_a - val_b)>0) return 1;
    else if((val_a - val_b)<0) return -1;
    else return 0;
}

int create_test_hosts_file
(
	const struct network_test_parameters_struct *parameters,
	char **hosts_names
)
{
	FILE *f;
	char *file_name;
	int i;

	file_name=(char *)malloc((strlen(parameters->file_name_prefix)+strlen("_hosts.txt")+1)*sizeof(char));
	if(file_name==NULL)
	{
		printf("Memory allocation error\n");
		return -1;
	}

	sprintf(file_name,"%s_hosts.txt",parameters->file_name_prefix);
	
	f=fopen(file_name,"w");
	if(f==NULL)
	{
		printf("File open error\n");
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

