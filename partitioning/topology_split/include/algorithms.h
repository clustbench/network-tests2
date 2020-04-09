#ifndef _ALGORITHMS_H_
#define _ALGORITHMS_H_

#include "net.h"

typedef struct Transmit
{
    char* sendNodeName;
    char* recvNodeName;

} Transmit;

typedef struct EqualityClass
{
    Transmit* listOfTransmissions;
    int transm;
    int hops;
} EqualityClass;

EqualityClass* findEqualClasses(Net* network, int* totalClasses);
void freeEqualityClasses(EqualityClass* cls, int totalClasses);

#endif