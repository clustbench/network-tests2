all:
	mpicc -I ../../common -I ../../core -fPIC -Wall -g -c all_to_all.c
	mpicc -g -shared -Wl,-soname,liball_to_all.so -o all_to_all.so all_to_all.o ../../common/benchmarks_common.o ../../core/my_time.o -lc

clean:
	rm -f *.o *.so
