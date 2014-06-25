#ifndef __TEST_NOISE_COMMON_H__
#define __TEST_NOISE_COMMON_H__

#include "my_time.h"
#include "tests_common.h"


extern int comm_rank;
extern int comm_size;

#define MODE_IDLE           0
#define MODE_GOAL_MESSAGES  1
#define MODE_NOISE_MESSAGES 2
#define MODE_FINISH_WORK    3


/*
 * Structure to keep most of test data together
 */
typedef struct tag_test_data
{
	px_my_time_type **tmp_results;
	char **send_data;
	char **recv_data;
	char **send_data_noise;
	char **recv_data_noise;
	int *processors;
} test_data;

#ifndef __TEST_NOISE_COMMON_C__

#ifdef __cplusplus
extern "C"
{
#endif

extern void init_test_data( test_data* td );
extern void clear_test_data( test_data* td );
extern void clear_more_test_data( test_data* td, int i );
extern int alloc_test_data( test_data* td, int mes_length, int num_repeats, int loading, int num_processors );
extern void free_test_data( test_data* td );
extern int random_choice( int proc1, int proc2, int num_processors, int* processors );

/**
 * This function fill mode_array on one MPI-proccess. The formed array contains data with all MPI-processes modes (Idle,Noise,Goal).
 * This array should be brodcasted to all MPI-processes.  
 */
extern int init_mode_array(int proc1,int proc2,int num_noise_procs,int num_all_procs,int *mode_array);

#ifdef __cplusplus
}
#endif

#endif /* __TEST_NOISE_COMMON_C__ */
#endif /* __TEST_NOISE_COMMON_H__   */

