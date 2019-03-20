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


all: liball.so libasync.so libput.so libget.so libcast2.so libnoiseblocking.so libnoise.so libnoisecommon.so libsend.so libone.so network_test print_from_to convert_to_netcdf

libput.so:  put2.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o put2.o tests_common.o -o libput.so

put2.o: put2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC put2.c -o put2.o

libget.so:  get2.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o get2.o tests_common.o -o libget.so

get2.o: get2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC get2.c -o get2.o


libcast2.so:  bcast2.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o bcast2.o tests_common.o -o libcast2.so

bcast2.o: bcast2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC bcast2.c -o bcast2.o


libnoiseblocking.so:  test_noise_blocking2.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o test_noise_blocking2.o tests_common.o -o libnoiseblocking.so

test_noise_blocking2.o: test_noise_blocking2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC test_noise_blocking2.c -o test_noise_blocking2.o


libnoise.so:  test_noise2.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o test_noise2.o tests_common.o -o libnoise.so

test_noise2.o: test_noise2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core/*.o -c -fPIC test_noise2.c -o test_noise2.o

libnoisecommon.so:  noisecommon.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o noisecommon.o tests_common.o -o libnoisecommon.so

noisecommon.o: test_noise_common.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC test_noise_common.c -o noisecommon.o

libsend.so:  send.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o send.o tests_common.o -o libsend.so

send.o: send_recv_and_recv_send2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC send_recv_and_recv_send2.c -o send.o


libone.so:  one.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared ../core/*.o one.o tests_common.o -o libone.so

one.o: one_to_one2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC one_to_one2.c -o one.o

liball.so:  all.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o all.o tests_common.o -o liball.so

all.o: all_to_all2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC all_to_all2.c -o all.o


libasync.so:  async.o parse_arguments.o tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../core/*.o async.o tests_common.o -o libasync.so

async.o: async_one_to_one2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -c -fPIC async_one_to_one2.c -o async.o

network_test: parse_arguments.o tests_common.o network_test2.o liball.so libasync.so libput.so libget.so libcast2.so libnoiseblocking.so libnoise.so libnoisecommon.so libsend.so libone.so
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) $(MPI_LIB_PATH) $(NETCDF_INCLUDE) -o network_test2 network_test2.o parse_arguments.o tests_common.o ../core/*.o -L. -lall -lasync -lput -lget -lcast2 -lnoiseblocking -lnoise -lnoisecommon -lsend -lone  -ldl -Wl,-rpath,. $(MPI_LIBS) -lnetcdf -lm

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



%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../core -I ../.. -c $^ -o $@

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../core -I ../.. -c $^ -o $@

all: 
	echo "Okay"
