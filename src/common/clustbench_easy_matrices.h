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
#ifndef __EASY_MATRICES_H__
#define __EASY_MATRICES_H__

#include <stdint.h>

//UINT нужен для обратной совместимости с C99?
typedef struct
{
	double *body;
	uint32_t sizex;
	uint32_t sizey;
} Clustbench_easy_matrix;

typedef struct
{
	double *body;
	uint32_t sizex;
	uint32_t sizey;
  uint32_t sizez;
} Clustbench_easy_matrix_3d;



#define MATRIX_GET_ELEMENT(matrix,x,y) *((matrix).body+(x)*(matrix).sizey+(y))

#define MATRIX_FILL_ELEMENT(matrix,x,y,elm)  *((matrix).body+(x)*(matrix).sizey+(y))=(elm)

#define MATRIX_GET_ELEMENT_3D(matrix,x,y,z)\
    *((matrix).body +                      \
      (x)*(matrix).sizez*(matrix).sizey +  \
      (y)*(matrix).sizez +                 \
      (z)                                  \
     )

#define MATRIX_FILL_ELEMENT_3D(matrix,x,y,z,elm)\
    *((matrix).body +                           \
      (x)*(matrix).sizez*(matrix).sizey +       \
      (y)*(matrix).sizez +                      \
      (z)                                       \
     ) = (elm)


#ifdef __cplusplus
extern "C"
{
#endif

int easy_mtr_create(Clustbench_easy_matrix *m, uint32_t x, uint32_t y);
int easy_mtr_create_3d(Clustbench_easy_matrix_3d *m, uint32_t x, uint32_t y, uint32_t z);

#ifdef __cplusplus
}
#endif

#endif /* __EASY_MATRICES_H__ */

