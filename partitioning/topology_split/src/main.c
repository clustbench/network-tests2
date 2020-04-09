#include <stdio.h>
#include <argp.h>
#include "net.h"
#include "algorithms.h"

char doc[] = "This tool finds equality classes in provided .gml network connectivity file. Output stored as csv file.";
char args_doc[] = "INPUT OUTPUT";

struct arguments {
    char *input;
    char *output;
};

void serializeEqualClasses(EqualityClass* classes, int totalClasses, char* filename)
{
    FILE* f = fopen(filename, "w");
    if (!f)
    {
        printf("Writing classes to file went wrong.\n");
        return;
    }

    for (int i = 0; i < totalClasses; ++i)
    {
        fprintf(f, "Class:Hops,SendNode,RecvNode\n");
        for (int j = 0; j < classes[i].transm; ++j)
        {
            fprintf(f, "%d:%d,%s,%s\n", i, classes[i].hops, classes[i].listOfTransmissions[j].sendNodeName, classes[i].listOfTransmissions[j].recvNodeName);
        }
    }
    fclose(f);

}

static error_t parse_option( int key, char* arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key)
    {
        case ARGP_KEY_ARG:
            if ( state->arg_num == 0) {
                arguments->input = arg;
            } else if ( state->arg_num == 1)
            {
                arguments->output = arg;
            } else {
                argp_usage( state );
            }
            break;
        case ARGP_KEY_END:
            if ( arguments->input == NULL || arguments->output == NULL)
            {
                argp_usage( state );
            }
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

struct argp argp = { NULL, parse_option, args_doc, doc};

int main(int argc, char** argv)
{
    struct arguments arguments = { 0 };
    argp_parse( &argp, argc, argv, 0 , 0, &arguments);
    Net* net = parseGML(arguments.input);

    if (!net)
        return 1;

    int totalClasses;
    EqualityClass* cls = findEqualClasses(net, &totalClasses);

    serializeEqualClasses(cls, totalClasses, arguments.output);

    freeNet(net);
    freeEqualityClasses(cls, totalClasses);

    return 0;
}