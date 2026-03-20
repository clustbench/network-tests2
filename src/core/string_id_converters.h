#ifndef __STRING_ID_CONVERTERS_H__
#define __STRING_ID_CONVERTERS_H__

/*
 * Data types
 */
#define NUM_NETWORK_TEST_DATATYPES 6

#define AVERAGE_NETWORK_TEST_DATATYPE     1
#define MEDIAN_NETWORK_TEST_DATATYPE      2
#define DEVIATION_NETWORK_TEST_DATATYPE   3
#define MIN_NETWORK_TEST_DATATYPE         4
#define ALL_DELAYS_NETWORK_TEST_DATATYPE 5
#define MEASUREMENTS_AMOUNT_TEST_DATATYPE 6

/*
 * Test types
 */

#define NUM_TEST_TYPES 11

#define UNKNOWN_TEST_TYPE 0
#define ONE_TO_ONE_TEST_TYPE 1
#define ALL_TO_ALL_TEST_TYPE 2
#define ASYNC_ONE_TO_ONE_TEST_TYPE 3
#define SEND_RECV_AND_RECV_SEND_TEST_TYPE 4
#define NOISE_TEST_TYPE 5
#define NOISE_BLOCKING_TEST_TYPE 6
#define BCAST_TEST_TYPE 7
#define PUT_ONE_TO_ONE_TEST_TYPE 8
#define GET_ONE_TO_ONE_TEST_TYPE 9
#define ONE_TO_ONE_CUDA_TEST_TYPE 10
#define ALL_TO_ALL_CUDA_TEST_TYPE 11

//это для того чтобы при компиляции на плюсах 
//стандарт вызова был под си?
#ifdef __cplusplus
extern "C"
{
#endif

//функция для вывода типа содержимого файла
//(среднее, медиана, ... )
extern const char *file_data_type_to_string(const int data_type);
//где эти описаны/используются?
extern int get_test_type(const char *str);
extern int get_test_type_name(int test_type,char *str);

#ifdef __cplusplus
}
#endif

#endif /* __STRING_ID_CONVERTERS_H__ */

