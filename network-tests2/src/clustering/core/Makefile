include ../../../config

data_netcdf.o: data_netcdf.cpp
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUUDE) $(NETCDF_INCLUDE) -c -o $@ $^

clean:
	rm -f *.o
