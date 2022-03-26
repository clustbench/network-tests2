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

#
# This file makes the interconnect_benchmark of the parus system
#

include ../../config

EXECS = interconnect_benchmark

FILES_IB =\
	benchmarks_main.o\
	get_node_name.o\
	parse_arguments.o


all: $(EXECS)

make_common: 
	$(MAKE) -C ../common

interconnect_benchmark: $(FILES_IB) make_common
	$(MPI_CCLINKER) $(MPI_CCLINKER_FLAGS) $(MPI_LIB_PATH) $(NETCDF_INCLUDE) -L ../core -L ../common -o $(EXECS) $(FILES_IB) ../common/clustbench_plugin_operations.o ../common/benchmarks_common.o ../common/clustbench_data_write_operations.o -lparus_network -lm -ldl $(MPI_LIBS)
	mv $(EXECS) $(INSTALL_DIR)
	
clean:
	rm  -f ./*.o
	rm  -f $(EXECS)
	$(MAKE) clean -C ../common
	
%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -ldl -I ../core -I ../.. -I ../common -c $^ -o $@
