include ../../../config

LIB = all_to_all

all:
	$(MPI_CC) $(NETCDF_INCLUDE) -I ../../common -I ../../core -O0 -fPIC -Wall -g -c -fpic $(LIB).c
	$(MPI_CCLINKER) $(NETCDF_INCLUDE) $(LIB_PATH) -g -O2 -shared -Wl,-soname,lib$(LIB).so -o $(LIB).so $(LIB).o ../../common/benchmarks_common.o ../common/delay_measurements_amount_auto.o ../../core/my_time.o -lc -fpic
	mkdir -p $(INSTALL_DIR)/lib
	mkdir -p $(INSTALL_DIR)/lib/clustbench-2.0.1
	mkdir -p $(INSTALL_DIR)/lib/clustbench-2.0.1/interconnect
	mkdir -p $(INSTALL_DIR)/lib/clustbench-2.0.1/interconnect/all_to_all
	cp $(LIB).so $(INSTALL_DIR)/lib/clustbench-2.0.1/interconnect/all_to_all

clean:
	rm -f *.o *.so
	rm -rf $(INSTALL_DIR)/lib/$(LIB)
