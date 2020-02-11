#ifndef DATA_ABSTRACT_H
#define DATA_ABSTRACT_H

#include <string>
#include <list>

using namespace std;

class Data_Abstract
{
public:
    typedef list<list<double> > matrix;

    inline string getSourceFileName()   {return source_file_name;}

    inline string getTestType()         {return test_type;}
    inline string getDataType()         {return data_type;}
    inline list<string> getHostNames()  {return host_names;}

    inline int getNumProcessors()       {return num_processors;}
    inline int getNumMessages()         {return num_messages;}
    inline int getBeginMessageLength()  {return begin_message_length;}
    inline int getEndMessageLength()    {return end_message_length;}
    inline int getStepLength()          {return step_length;}
    inline int getNoiseMessageLength()  {return noise_message_length;}
    inline int getNoiseMessageNum()     {return noise_message_num;}
    inline int getNoiseProcessors()     {return noise_processors;}
    inline int getNumRepeats()          {return num_repeats;}

    virtual int getRealEndMessageLength() = 0;
    virtual matrix getMatrix(int iMatrixLength) = 0;

protected:
    string source_file_name;
    //test parametres
    string test_type;
    string data_type;
    list<string> host_names;

    int num_processors;
    int num_messages; // Now it contains number of actually read matrices
    int begin_message_length;
    int end_message_length;
    int step_length;
    int noise_message_length;
    int noise_message_num;
    int noise_processors;
    int num_repeats;
};

#endif // DATA_ABSTRACT_H
