EXECS= delay_clustering

OBJS= ClusterAbstract.o ClusterData.o DBScan.o Divisive.o main.o ConfRoutine.o

LIBS= -lnetcdf -lhdf5

INC_DIR = headers
SRC_DIR = src

SOURCE= ClusterAbstract.cpp ClusterData.cpp DBScan.cpp Divisive.cpp main.cpp ConfRoutine.o
	
all: $(EXECS)
	
%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) -I$(INC_DIR) $(NETCDF_INCLUDE) -c -o $@  $^
 	
$(EXECS): $(OBJS)
	$(CXX) -o $(EXECS) $(OBJS) $(LIBS)
	
install: $(EXECS)
	cp -rf $^ $(INSTALL_DIR)/bin
	
clean:
	rm -f $(EXECS) *.o

.PHONY: clean
