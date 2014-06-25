#
# This file is a part of the PARUS project and  makes the core of the parus system
# Copyright (C) 2006  Alexey N. Salnikov (salnikov@cmc.msu.ru)
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

include ../../config


#FILES= demon.o edge_memory.o edge_data.o graph.o\
#       runer.o linealge.o simple_list.o px_log_print.o to_processor.o\
#       network_speed.o px_edges_list.o try_to_send.o str_operation.o\
#       my_time.o my_malloc.o

FILES=

#FILES_COMMON= linealge.o simple_list.o px_log_print.o\
#              network_speed.o my_time.o my_malloc.o

FILES_COMMON=network_speed.o\
	     my_time.o\
	     my_malloc.o\
	     linealge.o\
	     str_operation.o\
	     string_id_converters.o\
	     easy_matrices.o\
	     data_write_operations.o


              
EXECS= libparus_network.a

all: $(EXECS)
#	$(CCLINKER) -shared $(FILES) $(LIB_FLAGS)

core: $(FILES) $(FILES_COMMON)
	# Compile core OK!


libparus_network.a: $(FILES_COMMON)
	ar -r libparus_network.a $(FILES_COMMON)
	ranlib libparus_network.a


install: libparus_network.a
	cp -f *.h $(INSTALL_DIR)/include
	cp -f line_dynamic_array.cpp $(INSTALL_DIR)/include
	cp -f libparus_network.a $(INSTALL_DIR)/lib/libparus_network-$(PARUS_VERSION).a
	ln -s $(INSTALL_DIR)/lib/libparus_network-$(PARUS_VERSION).a $(INSTALL_DIR)/lib/libparus_network.a



clean:
	rm -f *.o libparus_network.a

.c.o:
	 $(MPI_cc) $(MPI_cc_FLAGS) $(MPI_cc_INCLUDE) $(NETCDF_INCLUDE) -I ../..  -c $*.c -o $*.o  

   
.cpp.o:
	 $(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) $(NETCDF_INCLUDE) -I ../..  -c $*.cpp -o $*.o
