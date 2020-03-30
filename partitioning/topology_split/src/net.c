#include "net.h"
#include <stdlib.h>

struct Unit
{
    int unitType;
    Unit** connectionList;

};

struct Net
{
    Unit** unitList;
    Unit** nodeList;
};

Net* parseGML(char* filename)
{
    return NULL;
}