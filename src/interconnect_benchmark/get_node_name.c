#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "clustbench_types.h"


int clustbench_get_node_name(char *node_name)
{
    if(gethostname(node_name, CLUSTBENCH_HOSTNAME_LENGTH) == -1)
    {
        fprintf(stderr,"clustbench_get_node_name: %s\n",strerror(errno));
        return -1;
    }
    
    return 0;
}

