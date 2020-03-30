#ifndef _ALGORITHMS_H_
#define _ALGORITHMS_H_

#include "net.h"

typedef struct EqualityClass
{
    char** inputNodes;
    char** outputNodes;

} EqualityClass;

void BellmanFord(Net* network);

#endif