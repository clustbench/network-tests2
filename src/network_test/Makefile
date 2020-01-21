
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

FILE_TESTS = \
	all_to_all2.o\
	async_one_to_one2.o\
	one_to_one2.o\
	send_recv_and_recv_send2.o\
	test_noise_common.o\
	test_noise2.o\
	test_noise_blocking2.o\
	bcast2.o\
	get2.o\
	put2.o\

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

FILES = one_to_one async_one_to_one put get all_to_all noise noise_blocking send_recv_and_recv_send bcast

EXECS= print_from_to convert_to_netcdf network_test2

all: one_to_one async_one_to_one put get all_to_all noise noise_blocking send_recv_and_recv_send bcast network_test2 print_from_to convert_to_netcdf

one_to_one: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/one_to_one
async_one_to_one: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/async_one_to_one
bcast: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/bcast
put: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/put
get: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/get
all_to_all: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/all_to_all
noise: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/noise
noise_blocking: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/noise_blocking
send_recv_and_recv_send: tests_common.o parse_arguments.o 
	$(MAKE) -C ./tests/send_recv_and_recv_send

#noisecommon.o: test_noise_common.c
#	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC test_noise_common.c -o noisecommon.o

network_test2: parse_arguments.o tests_common.o network_test2.o $(FILES)
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) $(MPI_LIB_PATH) $(NETCDF_INCLUDE) -rdynamic -o network_test2 network_test2.o parse_arguments.o tests_common.o ../core/*.o -ldl $(MPI_LIBS) -lm -lnetcdf -lconfig

network_test.o: network_test2.c
	$(MPI_CXX) $(MPI_CC_FLAGS) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) $(MPI_CC_INCLUDE) $(NETCDF_INCLUDE) -I ../core -c network_test2.c -ldl -o network_test2.o

print_from_to: $(FILES_PRINT_FROM_TO)
	$(CXXLINKER) $(CXXLINKER_FLAGS) $(MPI_LIB_PATH) $(NETCDF_INCLUDE) -L ../core -o print_from_to $(FILES_PRINT_FROM_TO) $(MPI_LIBS) -lnetcdf

convert_to_netcdf: convert_to_netcdf.o $(FILES_COMMON)
	$(MPI_CXXLINKER) $(MPI_CXXLINKER_FLAGS) $(MPI_LIB_PATH) -L ../core -o  convert_to_netcdf convert_to_netcdf.o ../core/string_id_converters.o  $(FILES_COMMON) $(MPI_LIBS) -lparus_network -lnetcdf


tests_common.o: tests_common.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c tests_common.c -o tests_common.o
	
parse_arguments.o: parse_arguments.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c parse_arguments.c -o parse_arguments.o -lconfig

all:
	echo "OK"

clean: clean_test
	rm  -f ./*.o
	rm -f ./*.so
	rm  -f $(EXECS) 

clean_test: 
	$(foreach rule, $(FILES), $(MAKE) clean -C ./tests/$(rule);)

install: all
	cp -rf $(EXECS) $(INSTALL_DIR)/bin/
	$(foreach hey, $(FILES), $(MAKE) install ./tests/$(hey);)

%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -I ../.. -c $^ -o $@

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../core -I ../.. -c $^ -o $@


