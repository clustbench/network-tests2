include ../../../../config

all: libsend_recv_and_recv_send.so

libsend_recv_and_recv_send.so:  send_recv_and_recv_send2.o ../../tests_common.o 
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -shared -L ../../../core/*.o send_recv_and_recv_send2.o ../../tests_common.o -o libsend_recv_and_recv_send.so

send_recv_and_recv_send2.o: send_recv_and_recv_send2.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -c -fPIC send_recv_and_recv_send2.c -o send_recv_and_recv_send2.o

clean:
	rm  -f ./*.o
	rm -f ./*.so
all:
	echo "Ok"

install: all
	cp -rf libsend_recv_and_recv_send.so $(INSTALL_DIR)/bin/

%.o: %.c
	$(MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../../../core -I ../.. -c $^ -o $@
