include ../../../config

LIB = all_to_all

all:
	mpicc -I ../../common -I ../../core -fPIC -Wall -g -c $(LIB).c
	mpicc -g -shared -Wl,-soname,lib$(LIB).so -o $(LIB).so $(LIB).o ../../common/benchmarks_common.o ../../core/my_time.o -lc
	mkdir $(INSTALL_DIR)/tests/$(LIB)
	cp $(LIB).so $(INSTALL_DIR)/tests/$(LIB)

clean:
	rm -f *.o *.so
	rm -rf $(INSTALL_DIR)/tests/$(LIB)
