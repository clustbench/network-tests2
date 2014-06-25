#ifndef __LINE_DYNAMIC_ARRAY_H__
#define __LINE_DYNAMIC_ARRAY_H__


#ifndef INTERNAL_ARRAY_ALONGATION_SIZE
	#define INTERNAL_ARRAY_ALONGATION_SIZE 2048
#endif

#ifndef INLINE
	#define INLINE
#endif

#define LINE_DYNAMIC_ARRAY_ERROR         -1
#define LINE_DYNAMIC_ARRAY_SUCCESS        0
#define LINE_DYNAMIC_ARRAY_ALREADY_EXISTS 1

/**
 * This class template implements the set abstraction as a regular array of 
 * pointers to elements.
 *
 * To use dynamic array is necessorary to define class Body_type that  
 * implements methods:
 * "int print()", "Body_type* copy()", "int equals(Body_type *element)"
 */
template <class Body_type>
class Line_dynamic_array
{
 protected:
 	
	/**
	 * The data stored in set.
	 */
	Body_type **data;
	
	/**
	 * The actual number of records (elements) in set.
	 */
	int num_records;

	/**
	 * This is the actual size of reserved elements 
	 * for data and keys.
	 *
	 * If there is no free space method add_element will 
	 * realloc the memory for data and keys.
	 */
	int current_internal_array_size;

public:
 	 
	 Line_dynamic_array();
	 ~Line_dynamic_array();
	 
	 /**
	  * Add new element to set.
	  *
	  * This method call method copy() of elements.
	  *
	  * Return:
	  *   LINE_DYNAMIC_ARRAY_SUCCESS if OK,
	  *   LINE_DYNAMIC_ARRAY_ERROR if error occurred,
	  *   LINE_DYNAMIC_ARRAY_ALREADY_EXISTS if the element with a specified 
	  *                              key already registered in set.
	  */
	 int add_element(Body_type *element);

	 /**
	  * This method search element in the 
	  * dynamic array. 
	  */
	 int find_element(Body_type *element);
	 
 
	 /**
	  * Remove element from position of dynamic array.
	  *
	  * Return:
	  *   0 if OK,
	  *  -1 if element is not found.
	  */
	 int delete_element(int position);
	
	 /**
	  * Return number of elements in array.
	  */
	 INLINE int num_elements(void)
	 {
		return num_records;
	 };

	 /**
	  * Printing the dynamic array to the standard output.
	  *
	  * This method call print() for keys and elements.
	  *
	  * Return 0 on success and -1 on error.
	  */
	 int print(void);

 
	 /**
	  * Search element by index in array. If this position is out 
	  * of range return NULL.
	  */
	 Body_type *look_position_uncopy(int position);

	/**
	  * Search element by index in array. If this position is out 
	  * of range return NULL.
	  */
	 Body_type *look_position(int position);


	 /**
	  * Return pointer to the internal array of data.
	  * Unsafe.
	  */
	 INLINE Body_type **get_elements_array(void)
	 {
	 	return data;
	 };

	 /**
	  * This code reduces size of internal arrays to size actual for 
	  * real number of records that are stored in this dynaic array.
	  *
	  * Return:
	  *  0 if ok
	  * -1 if error.
	  */
	 int vacuum(void);
};

#include "line_dynamic_array.cpp"

#undef INTERNAL_ARRAY_ALONGATION_SIZE

#endif /* end __LINE_DYNAMIC_ARRAY_H__ */

