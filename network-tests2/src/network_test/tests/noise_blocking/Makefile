include ../../../../config

all: libnoise_blocking.so

libnoise_blocking.so:  test_noise_blocking2.o ../../tests_common.o noisecommon.o
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../../../core/*.o test_noise_blocking2.o noisecommon.o ../../tests_common.o -o libnoise_blocking.so

test_noise_blocking2.o: test_noise_blocking2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -c -fPIC test_noise_blocking2.c -o test_noise_blocking2.o

noisecommon.o: test_noise_common.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -c -fPIC test_noise_common.c -o noisecommon.o


clean:
	rm  -f ./*.o
	rm -f ./*.so
all:
	echo "Ok"

install: all
	cp -rf libnoise_blocking.so $(INSTALL_DIR)/bin/

%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@
