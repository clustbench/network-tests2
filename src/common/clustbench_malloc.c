/*
 *  This file is a part of the CLUSTBENCH project.
 *  
 *  Authors:
 *      Alexey Salnikov (salnikov@cmc.msu.ru)
 *
 */

#include <stdlib.h>
/*
*************************************************************************
* This is redefined malloc because of some Operation Systems Like AIX   *
* does not allow allocate 0 bytes.                                      *
*                                                                       *
*************************************************************************
*/

//то есть другие файлы при попытке сделать что-то
//с созданной пустой памятью будут получать ошибку?
static char fake_memory;

void *__clustbench_malloc(size_t size)
{
 if (size==0) return &fake_memory;
 return malloc(size);
}

void __clustbench_free(void *pointer)
{
 if(pointer == &fake_memory) return;
 free(pointer);
 return;
}
