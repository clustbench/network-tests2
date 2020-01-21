#include "stdio.h"
#include "libconfig.h"
#include "stdlib.h"

int main()
{
	config_t cfg;
	config_setting_t *setting;
	const char *str;
	
	config_init(&cfg);

	if(! config_read_file(&cfg, "./proc_config.cfg"))
	{
		fprintf(stderr, "Error1\n");
		return(EXIT_FAILURE);
	}

	if (config_lookup_string(&cfg, "procs", &str))
		printf("%s\n", str);
	else
		printf("Error2\n");
	
	config_destroy(&cfg);
	return 0;

}
