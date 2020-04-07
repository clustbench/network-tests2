#ifndef _NET_H_
#define _NET_H_

typedef struct Unit Unit;
typedef struct Net Net;

typedef enum UnitType
{
    NODE, SWITCH
} UnitType;

typedef struct AdjNode
{
    int dest;
    struct AdjNode* next;
} AdjNode;

typedef struct AdjList 
{
    AdjNode* head;
} AdjList;

typedef struct Unit
{
    char* label;
    int id;
    UnitType type;
} Unit;


struct Net
{
    int nodes, switches;
    Unit* unitList;
    AdjList* adjLists;
};

Net* parseGML(char* filename);
void freeNet(Net* net);

#endif