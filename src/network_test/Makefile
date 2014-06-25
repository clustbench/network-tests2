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
# This file makes the nework_test of the parus system
#

include ../../config

FILES_COMMON=\
	tests_common.o

FILES_TEST2=\
	all_to_all2.o\
	async_one_to_one2.o\
	one_to_one2.o\
	send_recv_and_recv_send2.o\
	network_test2.o\
	test_noise_common.o\
	test_noise2.o\
	test_noise_blocking2.o\
	bcast2.o\
    	get2.o\
    	put2.o\
	parse_arguments.o

FILES_PRINT_FROM_TO=\
	print_from_to.o

	
               
EXECS= print_from_to convert_to_netcdf network_test2

all: $(EXECS)

network_test2: $(FILES_TEST2) $(FILES_COMMON)
	$(MPI_CLINKER) $(MPI_CLINKER_FLAGS) $(MPI_LIB_PATH) $(NETCDF_INCLUDE) -L ../core -o network_test2 $(FILES_COMMON) $(FILES_TEST2) $(MPI_LIBS) -lparus_network  -lnetcdf


print_from_to: $(FILES_PRINT_FROM_TO)
	$(CCLINKER) $(CCLINKER_FLAGS) $(MPI_LIB_PATH) $(NETCDF_INCLUDE) -L ../core -o print_from_to $(FILES_PRINT_FROM_TO) $(MPI_LIBS) -lnetcdf

convert_to_netcdf: convert_to_netcdf.o $(FILES_COMMON) 
	$(MPI_CCLINKER) $(MPI_CCLINKER_FLAGS) $(MPI_LIB_PATH) -L ../core -o  convert_to_netcdf convert_to_netcdf.o ../core/string_id_converters.o  $(FILES_COMMON) $(MPI_LIBS) -lparus_network -lnetcdf


clean:
	rm  -f ./*.o
	rm  -f $(EXECS)

install: all
	cp -rf $(EXECS) $(INSTALL_DIR)/bin/

%.o: %.c
	$(MPI_cc) $(MPI_cc_FLAGS) $(MPI_cc_INCLUDE) -I ../core -I ../.. -c $^ -o $@ 

   
%.o: %.cpp
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -I ../.. -c $^ -o $@

