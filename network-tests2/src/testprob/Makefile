binary: 
	network_test2.o network.a
	gcc -o binary network_test2.o -L. -lworld

main.o: 
	network_test2.c
	gcc -c main.c

libworld.a: 
	h_world.o g_world.o
	ar cr libworld.a h_world.o g_world.o

h_world.o: 
	h_world.c
	gcc -c h_world.c

g_world.o: 
	g_world.c
	gcc -c g_world.c

clean:
	rm -f *.o *.a binary
