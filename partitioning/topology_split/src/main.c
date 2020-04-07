#include <stdio.h>
#include "net.h"
#include "algorithms.h"

int main(int argc, char** argv)
{
    Net* net = parseGML(argv[1]);

    if (!net)
        return 1;

  //  BellmanFord(net);
    printf("%d %s\n", net->nodes, net->unitList[1].label);

    freeNet(net);


    return 0;
}