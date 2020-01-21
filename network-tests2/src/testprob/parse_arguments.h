#ifndef __PARSE_ARGUMENTS_H__
#define __PARSE_ARGUMENTS_H__

#include "types.h"

#define VERSION_FLAG 1
#define ERROR_FLAG   -1
#define HELP_FLAG    2

#ifdef __cplusplus
extern "C"
{
#endif

extern int parse_network_test_arguments(int argc,char **argv,struct network_test_parameters_struct *parameters);
extern int print_network_test_help_message(void);

#ifdef __cplusplus
}
#endif

#endif /* __PARSE_ARGUMENTS_H__ */
