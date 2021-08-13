/*
 * This file is part of clustbench project.
 * 
 *
 * Redefine malloc function if you need in it.
 * Reasoning to redefine may be if malloc(0) return NULL. 
 *
 * Authors:
 *  Alexey Salnikov (salnikov@cs.msu.ru)
 *
 */

#ifndef __CLUSTBENCH_MALLOC_H__
#define __CLUSTBENCH_MALLOC_H__

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern void *__clustbench_malloc(size_t size);
extern void __clustbench_free(void *pointer);

#ifdef __cplusplus
}
#endif

/* Uncomment it for redefine malloc
#define malloc __clustbench_malloc
#define free __clustbench_free

*/

#endif /* __CLUSTBENCH_MALLOC_H__ */

