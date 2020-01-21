include ../../../../config
all: liball_to_all.so

liball_to_all.so:  all.o ../../tests_common.o
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../../../core/*.o all.o ../../tests_common.o -o liball_to_all.so

all.o: all_to_all2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -c -fPIC all_to_all2.c -o all.o

clean:
	rm  -f ./*.o
	rm -f ./*.so
all:
	echo "Ok"

install: all
	cp -rf liball_to_all.so $(INSTALL_DIR)/bin/

%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@
