include ../../../../config

all: libone_to_one.so

libone_to_one.so:  one.o ../../tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../../../core/*.o one.o ../../tests_common.o -o libone_to_one.so

one.o: one_to_one2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -c -fPIC one_to_one2.c -o one.o

clean:
	rm  -f ./*.o
	rm -f ./*.so
all:
	echo "Ok"

install: all
	cp -rf libone_to_one.so $(INSTALL_DIR)/bin/

%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@
