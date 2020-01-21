lib =libcat.so libdog.so
#CFLAGS = -Wall -ansi -pedantic -shared -fPIC
#LDFLAGS = -ldl
#all: libdog.so libcat.so dlopen
all: libcat.so libdog.so dlopen

libcat.so: cat.o
	gcc -shared cat.o -o libcat.so

libdog.so: dog.o
	gcc -shared dog.o -o libdog.so

dog.o: dog.c
	gcc -c -fPIC dog.c -o dog.o

cat.o: cat.c
	gcc -c -fPIC cat.c -o cat.o

#%.o: dog.c cat.c
#	gcc -fPIC -c '$<' -o '$@'

dlopen: dlopen.o
	gcc -L. -ldog -lcat dlopen.o -ldl -o dlopen

dlopen.o: dlopen_sample.c
	gcc -c dlopen_sample.c -ldl -o dlopen.o

all:
	rm -rf *.o
	echo "All is the default"

