include ../../../config

LIB = all_to_all

all:
	$(MPI_CC) -I ../../common -I ../../core -O0 -fPIC -Wall -g -c $(LIB).c
	$(MPI_CCLINKER) -g -O2 -shared -Wl,-soname,lib$(LIB).so -o $(LIB).so $(LIB).o ../../common/benchmarks_common.o ../../core/my_time.o -lc
	mkdir $(INSTALL_DIR)/tests/$(LIB)
	cp $(LIB).so $(INSTALL_DIR)/tests/$(LIB)

clean:
	rm -f *.o *.so
	rm -rf $(INSTALL_DIR)/tests/$(LIB)
