
#ifndef __LINE_DYNAMIC_ARRAY_H__
 #error You must include 'line_dynamic_array.h' file before use this file
#else

#ifndef __LINE_DYNAMIC_ARRAY_CPP__
#define __LINE_DYNAMIC_ARRAY_CPP__

#include <stdlib.h>
#include <stdio.h>

/******************************************************************************/
template <class Body_type>
Line_dynamic_array<Body_type>::Line_dynamic_array()
{
	data=NULL;
	num_records=0;
	current_internal_array_size=0;
	return;
}
/******************************************************************************/
template <class Body_type>
Line_dynamic_array<Body_type>::~Line_dynamic_array()
{
	if(data!=NULL)
	{
		for(int i=0;i<num_records;i++)
		{
			if(data[i]!=NULL)
			{
				delete data[i];
				data[i]=NULL;
			}
		}
		free(data);
		data=NULL;
	}
	
	num_records=0;
	current_internal_array_size=0;
	return;
}
/******************************************************************************/
template <class Body_type>
int Line_dynamic_array<Body_type>::add_element
(
	Body_type *element
)
{
	if(num_records==current_internal_array_size)
	{
		Body_type **new_array=NULL;
	
		int new_array_size=current_internal_array_size+
		INTERNAL_ARRAY_ALONGATION_SIZE;
		
		new_array=(Body_type **)realloc(data,new_array_size*sizeof(Body_type *));
		if(new_array==NULL)
		{
			return LINE_DYNAMIC_ARRAY_ERROR;
		}

		data=new_array;
		current_internal_array_size=new_array_size;
	}

	Body_type *new_element=element->copy();
	if(new_element==NULL)
	{
		return LINE_DYNAMIC_ARRAY_ERROR;
	}

	data[num_records]=new_element;
	num_records++;
	
	return LINE_DYNAMIC_ARRAY_SUCCESS;
}
/******************************************************************************/
template <class Body_type>
int  Line_dynamic_array<Body_type>::delete_element(int position)
{
	if((position<0)||(position>=num_records))
	{
		return -1;
	}
	
	if(data[position]!=NULL)
	{
		delete data[position];	
	}
	
	num_records--;
	for(int i=position ; i<num_records ; i++)
	{
		data[i]=data[i+1];
	}

	return 0;
}
/******************************************************************************/
template <class Body_type>
int  Line_dynamic_array<Body_type>::print(void)
{
	printf("Line_dynamic_array content:\n\n");

	for(int i=0;i<num_records;i++)
	{
		data[i]->print();
		printf("\n");
	}

	return 0;
}
/******************************************************************************/
template <class Body_type>
Body_type* Line_dynamic_array<Body_type>::look_position_uncopy(int position)
{
	if((position<0) || (position>=num_records))
	{
		return NULL;
	}

	return data[position];
}
/******************************************************************************/
template <class Body_type>
Body_type* Line_dynamic_array<Body_type>::look_position(int position)
{
	if((position<0) || (position>=num_records))
	{
		return NULL;
	}

	return data[position]->copy();
}

/******************************************************************************/
template <class Body_type>
int Line_dynamic_array<Body_type>::vacuum(void)
{
	if(num_records==0)
	{
		if(data!=NULL)
		{
			free(data);
			data=NULL;
		}

		current_internal_array_size=0;
		return 0;
	}

	if(current_internal_array_size>num_records)
	{
	
		Body_type** new_data=(Body_type **)realloc
		(
			data,
			num_records*sizeof(Body_type *)
		);

		if(new_data==NULL)
		{
			return -1;
		}

		data=new_data;
	}

	return 0;
}
/******************************************************************************/
template <class Body_type>
int Line_dynamic_array<Body_type>::find_element(Body_type *pattern)
{
	int i;
	for(i=0;i<num_records;i++)
	{
		if(data[i]->equals(pattern))
		{
			return 1;
		}
	}
	return 0;
}
/******************************************************************************/
#endif /*  __LINE_DYNAMIC_ARRAY_CPP__ */

#endif /*  __LINE_DYNAMIC_ARRAY_H__   */

