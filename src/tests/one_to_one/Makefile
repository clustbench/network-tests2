include ../../../config

LIB = one_to_one

all:
	$(MPI_CC) $(NETCDF_INCLUDE) -I ../../common -I ../../core -O0 -fPIC -Wall -g -c -fpic $(LIB).c
	$(MPI_CCLINKER) $(NETCDF_INCLUDE) $(LIB_PATH) -g -O2 -shared -Wl,-soname,lib$(LIB).so -o $(LIB).so $(LIB).o ../../common/benchmarks_common.o ../../core/my_time.o -lc -fpic -fPIC
	mkdir -p $(INSTALL_DIR)/lib
	mkdir -p $(INSTALL_DIR)/lib/clustbench-2.0.1
	mkdir -p $(INSTALL_DIR)/lib/clustbench-2.0.1/interconnect
	mkdir -p $(INSTALL_DIR)/lib/clustbench-2.0.1/interconnect/one_to_one
	cp $(LIB).so $(INSTALL_DIR)/lib/clustbench-2.0.1/interconnect/one_to_one

clean:
	rm -f *.o *.so
	rm -rf $(INSTALL_DIR)/lib/$(LIB)
