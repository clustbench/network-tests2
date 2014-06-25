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
#ifndef __LINEALGE_H_
#define __LINEALGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <netcdfcpp.h>



#define Element double

#ifndef INLINE
    #define INLINE inline
#endif

class Vector
{
 protected:
  Element *body;
  int size;
 public:
  Vector();
  ~Vector();
  INLINE int get_size(void) { return size; }; 
  INLINE Element element(int number) 
  {
   if((number<0)||(number>=size)) return 0;
   return body[number];
  };
  int fread(FILE *f);
  INLINE Element *get_body(void) { return body; };
};



class Matrix
{
 protected:
  Element* body;
  int sizex;
  int sizey;
 public:
  Matrix();
  ~Matrix();
  Element element(int x, int y)
  {
   return *(body+x*sizey+y);
  };
  int fread(FILE *f,int x,int y);
  int mtr_create(int x, int y);
  //int write_netcdf(NcVar* data);
  INLINE void fill_element(int x,int y,Element elm)
  {
   *(body+x*sizey+y)=elm;
  };
  INLINE Element* get_body()
  {
     return body;
  }
};

#endif /* __LINEALGE_H_ */
