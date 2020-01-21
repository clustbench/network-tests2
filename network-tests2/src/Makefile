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

EXECS= print_from_to convert_to_netcdf network_test2
#-lall -lasync -lput -lget -lcast2 -lnoiseblocking -lnoise -lnoisecommon -lsend -lone

all: liball_to_all.so libasync_one_to_one.so libput.so libget.so libbcast.so libnoise_blocking.so libnoise.so libsend_recv_and_recv_send.so libone_to_one.so network_test2 print_from_to convert_to_netcdf

libput.so:  put2.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o put2.o tests_common.o -o libput.so

put2.o: put2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC put2.c -o put2.o

libget.so:  get2.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o get2.o tests_common.o -o libget.so

get2.o: get2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC get2.c -o get2.o


libbcast.so:  bcast2.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o bcast2.o tests_common.o -o libbcast.so

bcast2.o: bcast2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC bcast2.c -o bcast2.o


libnoise_blocking.so:  test_noise_blocking2.o parse_arguments.o tests_common.o noisecommon.o
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o noisecommon.o test_noise_blocking2.o tests_common.o -o libnoise_blocking.so

test_noise_blocking2.o: test_noise_blocking2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC test_noise_blocking2.c -o test_noise_blocking2.o


libnoise.so:  test_noise2.o parse_arguments.o tests_common.o noisecommon.o
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o noisecommon.o test_noise2.o tests_common.o -o libnoise.so

test_noise2.o: test_noise2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC test_noise2.c -o test_noise2.o


noisecommon.o: test_noise_common.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC test_noise_common.c -o noisecommon.o

libsend_recv_and_recv_send.so:  send.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o send.o tests_common.o -o libsend_recv_and_recv_send.so

send.o: send_recv_and_recv_send2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC send_recv_and_recv_send2.c -o send.o


libone_to_one.so:  one.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared ../core/*.o one.o tests_common.o -o libone_to_one.so

one.o: one_to_one2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC one_to_one2.c -o one.o

liball_to_all.so:  all.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o all.o tests_common.o -o liball.so

all.o: all_to_all2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC all_to_all2.c -o all.o


libasync_one_to_one.so:  async.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o async.o tests_common.o -o libasync_one_to_one.so

async.o: async_one_to_one2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC async_one_to_one2.c -o async.o

network_test2: parse_arguments.o tests_common.o network_test2.o 
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) $(MPI_LIB_PATH) $(NETCDF_INCLUDE) -rdynamic -o network_test2 network_test2.o parse_arguments.o tests_common.o ../core/*.o -ldl $(MPI_LIBS) -lm -lnetcdf

network_test.o: network_test2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) $(NETCDF_INCLUDE) -I ../core -c network_test2.c -ldl -o network_test2.o


print_from_to: $(FILES_PRINT_FROM_TO)
	$(CXXLINKER) $(CXXLINKER_FLAGS) $(MPI_LIB_PATH) $(NETCDF_INCLUDE) -L ../core -o print_from_to $(FILES_PRINT_FROM_TO) $(MPI_LIBS) -lnetcdf

convert_to_netcdf: convert_to_netcdf.o $(FILES_COMMON)
	$(MPI_CXXLINKER) $(MPI_CXXLINKER_FLAGS) $(MPI_LIB_PATH) -L ../core -o  convert_to_netcdf convert_to_netcdf.o ../core/string_id_converters.o  $(FILES_COMMON) $(MPI_LIBS) -lparus_network -lnetcdf


tests_common.o: tests_common.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c tests_common.c -o tests_common.o
	
parse_arguments.o: parse_arguments.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c parse_arguments.c -o parse_arguments.o

all:
	echo "OK"

clean:
	rm  -f ./*.o
	rm -f ./*.so
	rm  -f $(EXECS) 

%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -I ../.. -c $^ -o $@

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../core -I ../.. -c $^ -o $@


