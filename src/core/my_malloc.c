/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov (salnikov@cmc.msu.ru)
 *  
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
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

char fake_memory; 

void *__my_malloc(size_t size)
{
 if (size==0) return &fake_memory;
 return malloc(size);
}

void __my_free(void *pointer)
{
 if(pointer==&fake_memory) return;
 free(pointer);
 return;
}
