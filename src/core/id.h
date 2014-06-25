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
#ifndef __ID_H_
#define __ID_H_


#include <stdlib.h>
#include <stdio.h>

#include "types.h"

#ifndef INLINE
    #define INLINE inline
#endif

class ID
{
 protected:
  long_long_int id;
 public:
  ID(ID *cid) { id=cid->id; };
  ID() {id=-1; };
  ID(int cid) { id=cid; };
  ID(char cid) { id=cid; };
  ID(long cid) { id=cid; };
  ID(unsigned long cid) { id=cid; };
  //ID(long_long_int cid) { id=cid; };
  int print(void) { return printf("ID='%ld'\n",(long int)id); };
  INLINE ID *copy() 
    {
    ID *result;
    result=new ID;
    if(result==NULL) return NULL;
    result->id=id;
    return result;
   };
  INLINE operator long_long_int(void)  {return id;}; 
};

#endif /* __ID_H_ */
