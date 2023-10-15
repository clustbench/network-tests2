#ifndef __STRING_ID_CONVERTERS_H__
#define __STRING_ID_CONVERTERS_H__

/*
 * Data types
 */
#define NUM_NETWORK_TEST_DATATYPES 5

#define AVERAGE_NETWORK_TEST_DATATYPE     1
#define MEDIAN_NETWORK_TEST_DATATYPE      2
#define DEVIATION_NETWORK_TEST_DATATYPE   3
#define MIN_NETWORK_TEST_DATATYPE         4
#define ALL_DELAYS_NETWORK_TEST_DATATYPE 5

/*
 * Test types
 */


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

