#
# This file is a part of the PARUS project and  makes the core of the parus system
# Copyright (C) 2006  Alexey N. Salnikov
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
# Alexey N. Salnikov (salnikov@cmc.msu.ru)
#

include ../../config

FILES_COMMON =\
	benchmarks_common.o\
	clustbench_data_write_operations.o\
	clustbench_easy_matrices.o\
	clustbench_malloc.o\
	clustbench_plugin_operations.o\
	clustbench_time.o\
	delay_measurements_amount_auto.o
	
all: $(FILES_COMMON)

clean: 
	rm  -f ./*.o
	
%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) $(NETCDF_INCLUDE) -fPIC -ldl -I ../core -I ../.. -I ../common -c $^ -o $@
