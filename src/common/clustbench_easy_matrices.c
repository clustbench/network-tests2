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
#include <stdint.h>

#include "clustbench_easy_matrices.h"

int  easy_mtr_create(Clustbench_easy_matrix *m, uint32_t x, uint32_t y)
{
	m->sizex=x;
	m->sizey=y;
	m->body=(double *)malloc(x*y*sizeof(double));
	   if(m->body==NULL) return -1;
	return 0;
}

int  easy_mtr_create_3d(Clustbench_easy_matrix_3d *m, uint32_t x, uint32_t y, uint32_t z)
{
	m->sizex=x;
	m->sizey=y;
        m->sizez=z;
	m->body=(double *)malloc(x*y*z*sizeof(double));
	   if(m->body==NULL) return -1;
	return 0;
}

