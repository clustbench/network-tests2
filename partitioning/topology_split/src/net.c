#include "net.h"

typedef enum ParserState
{
    BEGIN, IN_GRAPH, PARSE_GRAPH, IN_NODE, PARSE_NODE, IN_LINK, PARSE_LINK, END, GRAPHICS_SECTION
} ParserState;

typedef enum LinkDirect
{
    BI, ONE
} LinkDirect;

Net* createNet(int nodes, int switches)
{
    Net* ret = (Net*)malloc(sizeof(Net));
    ret->switches = switches;
    ret->nodes = nodes;
    ret->unitList = (Unit*)malloc(sizeof(Unit) * (nodes + switches + 1));
    ret->adjLists = (AdjList*)malloc(sizeof(AdjList) * (nodes + switches + 1));
    for (int i = 0; i < nodes + switches + 1; ++i)
    {
        ret->adjLists[i].head = NULL;
    }

    return ret;    
}

void setUnit(Net* net, Unit *unit)
{
    int id = unit->id;
    net->unitList[id].id = id;
    net->unitList[id].type = unit->type;
    net->unitList[id].label = (char*)malloc((strlen(unit->label) + 1) * sizeof(char));
    strncpy(net->unitList[id].label, unit->label, strlen(unit->label));
    net->unitList[id].label[strlen(unit->label)] = 0;
}

void addLink(Net* net, int source, int destination)
{
    AdjNode *head = net->adjLists[source].head;

    AdjNode *newNode = (AdjNode*)malloc(sizeof(AdjNode));
    
    newNode->next = head;
    newNode->dest = destination;
    net->adjLists[source].head = newNode;

}

void freeNet(Net* net)
{
    for (int i = 0; i < net->nodes + net->switches + 1; ++i)
    {
        free(net->unitList[i].label);

        AdjNode *tmp_head = net->adjLists[i].head;
        AdjNode *next = NULL;
        while(tmp_head != NULL)
        {
            next = tmp_head->next;
            free(tmp_head);
            tmp_head = next;
        }
    }
    free(net->unitList);
}

void cleanToken(char* token, int* k)
{
    for (int i = 0; i < *k; ++i)
    {
        token[i] = 0;
    }
    *k = 0;
}

char* getParam(FILE *f)
{
    char c;
    c = fgetc(f);
    char* tok = (char*)malloc(sizeof(char) * 255);
    memset(tok, '\0', 255);
    int k = 0;
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\"')
    {
        c = fgetc(f);
    }
    while (!(c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\"'))
    {
        tok[k] = c;
        k++;
        c = fgetc(f);
    }
    return tok;
}

Net* parseGML(char* filename)
{
    Net* resultNet = NULL;
    FILE *f = NULL;
    char tok[255];
    memset(tok, '\0', 255);
    int nodes = -1;
    int switches = -1;
    f = fopen(filename, "r");
    if (f == NULL)
    {
        printf("Unable read file!\n");
        return NULL;
    }
    int k = 0;
    ParserState state = BEGIN;
    ParserState prevState = BEGIN;
    Unit *unIns;
    char c;
    int source = -1;
    int target = -1;
    LinkDirect linkType;
    while ((c = fgetc(f)) != EOF && state != END)
    {
        if (nodes != -1 && switches != -1 && resultNet == NULL)
        {
            resultNet = createNet(nodes, switches);
        }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        {
            switch (state)
            {
                case BEGIN:
                    if (!strncmp(tok, "graph", 255))
                    {
                        state = IN_GRAPH;
                    }
                    cleanToken(tok, &k);
                    break;
                case IN_GRAPH:
                    if (!strncmp(tok, "[", 255))
                    {
                        state = PARSE_GRAPH;
                    } 
                    else if (!strncmp(tok, "]", 255))
                    {
                        state = END;
                    }
                    cleanToken(tok, &k);
                    break;
                case PARSE_GRAPH:
                    if (!strncmp(tok, "nodesNum", 255))
                    {
                        char* param = getParam(f);
                        nodes = atoi(param);
                        free(param);
                    } 
                    else if (!strncmp(tok, "switchesNum", 255))
                    {
                        char* param = getParam(f);
                        switches = atoi(param);
                        free(param);
                    } 
                    else if (!strncmp(tok, "node", 255))
                    {
                        state = IN_NODE;
                    } 
                    else if (!strncmp(tok, "edge", 255))
                    {
                        state = IN_LINK;
                    }
                    
                    else if (!strncmp(tok, "]", 255))
                    {
                        state = END;
                    } 
                    cleanToken(tok, &k);
                    break;
                case IN_NODE:
                    if (!strncmp(tok, "[", 255))
                    {
                        size_t q = sizeof(Unit);
                        unIns = (Unit*)malloc(q);
                        unIns->label = NULL;
                        state = PARSE_NODE;
                    }
                    cleanToken(tok, &k);
                    break;
                case PARSE_NODE:
                    if (!strncmp(tok, "id", 255))
                    {
                        char* param = getParam(f);
                        unIns->id = atoi(param);
                        free(param);
                    } 
                    else if (!strncmp(tok, "label", 255))
                    {
                        char* param = getParam(f);
                        if (unIns->label != NULL)
                            free(unIns->label);
                        size_t strl = strlen(param);
                        unIns->label = (char*)malloc((strl + 1) * sizeof(char));
                        strncpy(unIns->label, param, strl);
                        unIns->label[strl] = 0;
                        free(param);
                    } 
                    else if (!strncmp(tok, "deviceType", 255))
                    {
                        char* param = getParam(f);
                        if (!strncmp(param, "switch", 255))
                        {
                            unIns->type = SWITCH;
                        }
                        if (!strncmp(param, "node", 255))
                        {
                            unIns->type = NODE;
                        }
                        free(param);
                    } 
                    else if (!strncmp(tok, "graphics", 255)) { 
                        prevState = state;
                        state = GRAPHICS_SECTION;
                    }
                    else if (!strncmp(tok, "]", 255))
                    {
                        setUnit(resultNet, unIns);
                        free(unIns);
                        state = PARSE_GRAPH;
                    }
                    cleanToken(tok, &k);
                    break;
                case IN_LINK:
                    if (!strncmp(tok, "[", 255))
                    {
                        state = PARSE_LINK;
                    }
                    cleanToken(tok, &k);
                    break;
                case PARSE_LINK:
                    if (!strncmp(tok, "source", 255))
                    {
                        char* param = getParam(f);
                        source = atoi(param);
                        free(param);
                    } 
                    else if (!strncmp(tok, "target", 255))
                    {
                        char* param = getParam(f);
                        target = atoi(param);
                        free(param);
                    }
                    else if (!strncmp(tok, "graphics", 255))
                    {
                        prevState = state;
                        state = GRAPHICS_SECTION;
                    } 
                    else if (!strncmp(tok, "directed", 255))
                    {
                        char* param = getParam(f);
                        if (!strncmp(param, "BI", 255))
                        {
                            linkType = BI;
                        }
                        if (!strncmp(param, "ONE", 255))
                        {
                            linkType = ONE;
                        }
                        free(param);
                    }
                    else if (!strncmp(tok, "]", 255))
                    {
                        addLink(resultNet, source, target);
                        if (linkType == BI)
                        {
                            addLink(resultNet, target, source);
                        }
                        state = PARSE_GRAPH;
                    }
                    cleanToken(tok, &k);
                    break;
                case GRAPHICS_SECTION:
                    if (!strncmp(tok, "]", 255))
                    {
                        state = prevState;
                    }
                    cleanToken(tok, &k);
                    break;
                case END:
                default:
                    cleanToken(tok, &k);
                    break;
            }
        } else 
        {
            tok[k] = c;
            k++;
        }
    }
    fclose(f);
    return resultNet;
}