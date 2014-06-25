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

#ifndef __MY_TYPES_H__
#define __MY_TYPES_H__

/* #include "id.h" */

#define Key_type ID
/* typedef int long_long_int; */
#define long_long_int int 

struct network_test_parameters_struct
{
	int  num_procs;
	int  test_type;
	int  begin_message_length;
	int  end_message_length;
	int  step_length;
	int  num_repeats;
	int  noise_message_length;
	int  num_noise_messages;
	int  num_noise_procs;
	const char *file_name_prefix;
};


#endif /* __MY_TYPES_H__ */
