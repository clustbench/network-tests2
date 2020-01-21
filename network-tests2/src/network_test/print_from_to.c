#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netcdf.h>

#define BUF_SIZE 1024

int main(int argc,char **argv)
{

	FILE *hosts_file;

	char from_host[BUF_SIZE] = "none";
	char to_host[BUF_SIZE]   = "none";
	char tmp_str[BUF_SIZE];

	int from_proc;
	int to_proc;

	size_t i;
	
	int flag;

	int netcdf_file_id;

	int x_dim_id, y_dim_id, num_matrices_dim_id;
	size_t x,y,n;
	int data_var_id;
	int begin_mes_length_var_id, step_length_var_id;

	int begin_message_length;
	int step_message_length;
	long message_length;

	size_t start[3],count[3];

	double *data=NULL;

	if(argc < 4)
	{
		fprintf(stderr,"\n\n\tprint_from_to <send proc> <recv proc> <netcdf file with network_test data> [ <hosts file> ]\n\n\n");
		return -1;		
	}

	from_proc=atoi(argv[1]);
	to_proc=atoi(argv[2]);

	flag=nc_open(argv[3],NC_NOWRITE,&netcdf_file_id);	
	if(flag)
	{
		fprintf(stderr,"Can't read file with name '%s' which contains data genereted by network_test\n",argv[3]);
		return -1;
	}

	flag=nc_inq_dimid(netcdf_file_id,"x",&x_dim_id);
	if(flag)
	{
		fprintf(stderr,"x dimension not available\n");
		return -1;
	}

	flag=nc_inq_dimid(netcdf_file_id,"y",&y_dim_id);
	if(flag)
	{
		fprintf(stderr,"y dimension not available\n");
		return -1;
	}

	flag=nc_inq_dimid(netcdf_file_id,"n",&num_matrices_dim_id);
	if(flag)
	{
		fprintf(stderr,"n dimension not available\n");
		return -1;
	}

	nc_inq_dimlen(netcdf_file_id,x_dim_id,&x);
	nc_inq_dimlen(netcdf_file_id,y_dim_id,&y);
	nc_inq_dimlen(netcdf_file_id,num_matrices_dim_id,&n);

	flag=nc_inq_varid(netcdf_file_id,"begin_mes_length",&begin_mes_length_var_id);
	if(flag)
	{
		fprintf(stderr,"Variable 'begin_mes_length' is not avaibale in dataset\n");
		return -1;
	}

	
	flag=nc_inq_varid(netcdf_file_id,"step_length",&step_length_var_id);
	if(flag)
	{
		fprintf(stderr,"Variable 'step_length' is not avaibale in dataset\n");
		return -1;
	}

	flag=nc_inq_varid(netcdf_file_id,"data",&data_var_id);
	if(flag)
	{
		fprintf(stderr,"Variable 'data' is not avaibale in dataset\n");
		return -1;
	}

	if(from_proc > x)
	{
		fprintf(stderr,"Process with number %d that is sending data is out of processor set.\n",from_proc);
		fprintf(stderr,"Available only %ld rocessors.\n",(long int)x);
		return -1;
	}

	if(to_proc > y)
	{
		fprintf(stderr,"Process with number %d that is recieveing data is out of processor set.\n",to_proc);
		fprintf(stderr,"Available only %ld processors.\n",(long int)y);
		return -1;
	}

	flag=nc_get_var_int(netcdf_file_id,begin_mes_length_var_id,&begin_message_length);
	if(flag)
	{
		fprintf(stderr,"can't read data from variable '%s'\n","begin_mes_length");
		return -1;
	}



	flag=nc_get_var_int(netcdf_file_id,step_length_var_id,&step_message_length);
	if(flag)
	{
		fprintf(stderr,"can't read data from variable '%s'\n","step_length");
		return -1;
	}

	if(argc>=5)
	{

		int max=(from_proc>to_proc)?from_proc:to_proc;

		hosts_file=fopen(argv[4],"rt");
		if(hosts_file==NULL)
		{
			fprintf(stderr,"Can't open file '%s'\n",argv[4]);
			return -1;
		}
		

		for(i=0;i<=max;i++)
		{
			if(fgets(tmp_str,BUF_SIZE,hosts_file)==NULL)
			{
				fprintf(stderr,"Unexpected end of file in '%s'\n",argv[4]);
				return -1;
			}

			if(i==from_proc)
			{
				strcpy(from_host,tmp_str);
				from_host[strlen(from_host)-1]='\0';
			}

			if(i==to_proc)
			{
				strcpy(to_host,tmp_str);
				to_host[strlen(to_host)-1]='\0';
			}

		}

		fclose(hosts_file);
	}

	data=(double *)malloc(n*sizeof(double));
	if(data==NULL)
	{
		fprintf(stderr,"Can't allocate %ld bytes for data from dataset",(long int)(n*sizeof(double)));
		return -1;
	}

	start[0]=0;
	start[1]=from_proc;
	start[2]=to_proc;

	count[0]=n;
	count[1]=1;
	count[2]=1;

	flag=nc_get_vara_double(netcdf_file_id,data_var_id,start,count,data);
	if(flag)
	{
		fprintf(stderr,"can't read data from variable '%s'\n","data");
		return -1;
	}

	nc_close(netcdf_file_id);

	printf
	(
	 	"Delays during data transfers from %d (%s) to %d (%s) process\n\nlength\tvalue\n",
		from_proc,from_host,
		to_proc,to_host
	);
	
	for(i=0;i<n;i++)
	{
		message_length=begin_message_length+i*step_message_length;
		printf("%ld\t%12.12f\n",message_length,(double)data[i]);
	}

	free(data);
	
	return 0;
}

