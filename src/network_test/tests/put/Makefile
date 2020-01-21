include ../../../../config

all: libput.so

libput.so:  put2.o ../../tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../../../core/*.o put2.o ../../tests_common.o -o libput.so

put2.o: put2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -c -fPIC put2.c -o put2.o

clean:
	rm  -f ./*.o
	rm -f ./*.so
all:
	echo "Ok"

install: all
	cp -rf libput.so $(INSTALL_DIR)/bin/

%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@
