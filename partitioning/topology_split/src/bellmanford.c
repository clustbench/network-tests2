#include "net.h"
#include "algorithms.h"

const int INF = 10000000;

typedef struct Edge
{
    int source;
    int destination;
    int cost;
} Edge;

typedef struct EdgeList
{
    Edge* edges;
    int cnt;
} EdgeList;

typedef struct intVector
{
    int* data;
    int size;
} intVector;

intVector findCoreOutSwitches(Net* network)
{
    intVector coreSwitches;
    coreSwitches.data = NULL;
    coreSwitches.size = 0;
    for (int i = 0; i < network->nodes; ++i)
    {
        AdjNode *head = network->adjLists[i].head;
        while (head != NULL)
        {
            if (network->unitList[head->dest].type == SWITCH) 
            {
                int was_added = 0;
                for (int i = 0; i < coreSwitches.size; ++i)
                {
                    if (head->dest == coreSwitches.data[i])
                        was_added = 1;
                }
                if (!was_added)
                {
                    if (coreSwitches.size == 0)
                    {
                        coreSwitches.data = (int*)malloc(sizeof(int));
                        coreSwitches.size++;
                    } else {
                        coreSwitches.data = (int*)realloc((void*)coreSwitches.data, sizeof(int) * (coreSwitches.size + 1));
                        coreSwitches.size++;
                    }
                    coreSwitches.data[coreSwitches.size - 1] = head->dest;
                }
            }
            head = head->next;
        }
    }
    return coreSwitches;
}


intVector* findinNodesSwitches(Net* network)
{
    intVector* inNodesSwitches;
    inNodesSwitches = (intVector*)malloc(sizeof(intVector) * (network->nodes + 1));

    for (int i = 0; i <= network->nodes; ++i)
    {
        inNodesSwitches[i].data = NULL;
        inNodesSwitches[i].size = 0;
    }

    for (int i = 1; i <= network->switches; ++i)
    {
        AdjNode *head = network->adjLists[network->nodes + i].head;
        while (head != NULL)
        {
            if (network->unitList[head->dest].type == NODE) 
            {
                if (inNodesSwitches[head->dest].size == 0)
                {
                    inNodesSwitches[head->dest].data = (int*)malloc(sizeof(int));
                    inNodesSwitches[head->dest].size++;
                } else {
                    inNodesSwitches[head->dest].data = (int*)realloc((void*)inNodesSwitches[head->dest].data, sizeof(int) * (inNodesSwitches[head->dest].size + 1));
                    inNodesSwitches[head->dest].size++;
                }
                inNodesSwitches[head->dest].data[inNodesSwitches[head->dest].size - 1] = i + network->nodes;
            }
            head = head->next;
        }
    }

    return inNodesSwitches;
}

int** BellmanFord(EdgeList* switchNetwork, intVector* coreSwitches, int switches, int nodes)
{
    int** result = (int**)malloc(sizeof(int*) * coreSwitches->size);
    for (int i = 0; i < coreSwitches->size; ++i)
    {
        result[i] = (int*)malloc(sizeof(int) * switches);
    }
    for (int i = 0;  i < coreSwitches->size; ++i)
    {
        for (int j = 0; j < switches; ++j)
        {
            if ((coreSwitches->data[i] - nodes - 1) == j) 
            {
                result[i][j] = 0;
            } 
            else 
            {
                result[i][j] = INF;
            }
        }
    }
    int changed = 0;
    for (int v = 0; v < coreSwitches->size; ++v)
    {
        while(1) 
        {
            int changed = 0;
            for (int i = 0; i < switchNetwork->cnt; ++i)
            {
                if (result[v][switchNetwork->edges[i].source] < INF) 
                {
                    if (result[v][switchNetwork->edges[i].destination] > result[v][switchNetwork->edges[i].source] + switchNetwork->edges[i].cost)
                    {
                        result[v][switchNetwork->edges[i].destination] = result[v][switchNetwork->edges[i].source] + switchNetwork->edges[i].cost;
                        changed = 1;
                    }
                }
            }   
            if (!changed)
                break;
        }
    }
    return result;
}

int** getNodeDists(Net* network, intVector* inNodesSwitches, intVector* coreSwitches, int** coreDists, int nodes)
{
    int** nodeDists = (int**)malloc(sizeof(int*) * (nodes + 1));
    for (int i = 0; i <= nodes; ++i)
    {
        nodeDists[i] = (int*)malloc(sizeof(int) * (nodes + 1));
    }

    for (int i = 1; i <= nodes; ++i)
    {
        for (int j = 1; j <= nodes; ++j)
        {
            if (i == j)
                nodeDists[i][j] = 0;
            else 
            {
                int dist = INF;
                AdjNode* head = network->adjLists[i].head;
                while (head != NULL)
                {
                    if (network->unitList[head->dest].type == SWITCH)
                    {
                        int outidx = 0;
                        for (outidx; head->dest != coreSwitches->data[outidx]; ++outidx);
                        
                        for (int m = 0; m < inNodesSwitches[j].size; ++m)
                        {
                            int inidx = inNodesSwitches[j].data[m] - nodes - 1;
                            if (dist > (coreDists[outidx][inidx] + 2))
                            {
                                dist = coreDists[outidx][inidx] + 2;
                            }
                        }
                    }
                    head = head->next;
                }
                nodeDists[i][j] = dist;
            }
        }
    }
    return nodeDists;
}

EqualityClass* findEqualClasses(Net* network, int* totalClasses)
{
    EqualityClass* result = NULL;
    *totalClasses = 0;
    intVector coreSwitches = findCoreOutSwitches(network);
    intVector* inNodeSwitches = findinNodesSwitches(network);
    EdgeList graph; 
    graph.cnt = 0;
    graph.edges = NULL;
    for (int i = 1; i <= network->switches; ++i)
    {
        AdjNode* head = network->adjLists[i + network->nodes].head;
        while (head != NULL)
        {
            if (network->unitList[head->dest].type == SWITCH)
            {
                if (graph.cnt == 0)
                {
                    graph.edges = (Edge*)malloc(sizeof(Edge));
                    graph.cnt++;
                } else {
                    graph.edges = (Edge*)realloc((void*)graph.edges, sizeof(Edge) * (graph.cnt + 1));
                    graph.cnt++;
                }
                graph.edges[graph.cnt - 1].destination = head->dest;
                graph.edges[graph.cnt - 1].source = i + network->nodes;
                graph.edges[graph.cnt - 1].cost = 1;
            }
            head = head->next;
        }
    }
    for (int i = 0; i < graph.cnt; ++i)
    {
        graph.edges[i].source -= (network->nodes + 1);
        graph.edges[i].destination -= (network->nodes + 1);
    }

    int** switchDists = BellmanFord(&graph, &coreSwitches, network->switches, network->nodes);
    int** nodeDists = getNodeDists(network, inNodeSwitches, &coreSwitches, switchDists, network->nodes);
    
    for (int i = 1; i < network->nodes; ++i)
    {
        for (int j = 1; j < network->nodes; ++j)
        {
            if (i != j)
            {
                if (*totalClasses == 0)
                {
                    result = (EqualityClass*)malloc(sizeof(EqualityClass));
                    result[0].hops = nodeDists[i][j];
                    result[0].transm = 0;
                    (*totalClasses)++;
                }
                int clsIdx = -1;
                for (int k = 0; k < *totalClasses; ++k)
                {
                    if(nodeDists[i][j] == result[k].hops)
                        clsIdx = k;
                }
                if (clsIdx == -1)
                {
                    result = (EqualityClass*)realloc(result, sizeof(EqualityClass) * (*totalClasses + 1));
                    result[(*totalClasses)].hops = nodeDists[i][j];
                    clsIdx = (*totalClasses);
                    (*totalClasses)++;
                    
                }
                if (!result[clsIdx].transm)
                {
                    result[clsIdx].listOfTransmissions = (Transmit*)malloc(sizeof(Transmit));
                    result[clsIdx].transm++;
                } 
                else 
                {
                    result[clsIdx].listOfTransmissions = (Transmit*)realloc(result[clsIdx].listOfTransmissions, sizeof(Transmit) * 
                                                                             (result[clsIdx].transm + 1));
                    result[clsIdx].transm++;
                }
                result[clsIdx].listOfTransmissions[result[clsIdx].transm - 1].sendNodeName = (char*)malloc(sizeof(char) * (strlen(network->unitList[i].label) + 1));
                result[clsIdx].listOfTransmissions[result[clsIdx].transm - 1].recvNodeName = (char*)malloc(sizeof(char) * (strlen(network->unitList[j].label) + 1));
                memset(result[clsIdx].listOfTransmissions[result[clsIdx].transm - 1].sendNodeName, 0, strlen(network->unitList[i].label) + 1);
                strcpy(result[clsIdx].listOfTransmissions[result[clsIdx].transm - 1].sendNodeName, network->unitList[i].label);
                memset(result[clsIdx].listOfTransmissions[result[clsIdx].transm - 1].recvNodeName, 0, strlen(network->unitList[j].label) + 1);
                strcpy(result[clsIdx].listOfTransmissions[result[clsIdx].transm - 1].recvNodeName, network->unitList[j].label);
            }
        }
    }
    for (int i = 0; i <= coreSwitches.size; ++i)
    {
        free(switchDists[i]);
    }
    free(switchDists);
    free(coreSwitches.data);
    for (int i = 0 ; i <= network->nodes; ++i)
    {
        free(inNodeSwitches[i].data);
        free(nodeDists[i]);
    }
    free(inNodeSwitches);
    free(nodeDists);
    free(graph.edges);


    return result;
}

void freeEqualityClasses(EqualityClass* cls, int totalClasses)
{
    for(int i = 0; i < totalClasses; ++i)
    {
        for (int t = 0; t < cls[i].transm; t++)
        {
            free(cls[i].listOfTransmissions[t].recvNodeName);
            free(cls[i].listOfTransmissions[t].sendNodeName);
        }
        free(cls[i].listOfTransmissions);
    }
    free(cls);
}