/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov (salnikov@cmc.msu.ru)
 *  
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 * 
 */

#include "network_speed.h"
#include "str_operation.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "my_malloc.h"

#define READ_STR_LENGTH 300

/***********************************************************************/
Network_speed::Network_speed()
{
    state = info_state_nothing;

    num_processors=0;
    num_messages=0;

    /* Test parameters */
    test_type[0] = 0;
	data_type[0] = 0;
    begin_message_length = 0;
    end_message_length = 0;
    step_length = 0;
    noise_message_length = 0;
    noise_message_num = 0;
    noise_processors = 0;
    num_repeats = 0;
    host_names = 0;
    host_ranks = 0;

    messages_length=NULL;
    links=NULL;
    return;
}
/***********************************************************************/
            
Network_speed::~Network_speed()
{

	if ( host_names != NULL )
    {
        for ( int i = 0; i < num_processors; i++ )
		{
			if(host_names[i]!=NULL)
			{
				free( host_names[i] );
				host_names[i]=0;
			}
		}
        free( host_names );
		host_names=NULL;
    }

    if ( host_ranks != NULL )
	free( host_ranks );

    num_processors=0;
    num_messages=0;

    if(messages_length!=NULL)
        free(messages_length);
    if(links!=NULL)
        delete [] links;
    return;
}
/****************************************************************************/
int Network_speed::fread(char *file_name)
{
    FILE *f;
    int flag;
    char str[READ_STR_LENGTH];
    int i;
    state = info_state_no_file;
    f=fopen(file_name,"r");
    if(f==NULL)
    {
        printf("Network_speed::fread(char *) can not open file '%s'\n",file_name);
        return -1;
    }
    flag=get_word(f,str,READ_STR_LENGTH);
    if(flag)
        return -1;
    if(strcmp(str,"processors"))
    {
        printf("Bad format file %s\n",file_name);
        printf(" 'processor' record are not present '%s'\n",str);
        return -1;
    }

    flag=get_word(f,str,READ_STR_LENGTH);
    if(flag)
        return -1;
    num_processors=atoi(str);
    // Update state
    state = info_state_processors;

    /* After revision 61 file header contains only processor number
     * 
     * flag=get_word(f,str);
     * if(flag==-1) return -1;
     * if(strcmp(str,"num"))
     * {
     *  printf("Bad format file %s\n",file_name);
     *  printf(" record 'num' not precent '%s'\n",str);
     *  return -1;
     * }
     * 
     * flag=get_word(f,str);
     * if(flag==-1) return -1;
     * if(strcmp(str,"messages"))
     * {
     *  printf("Bad format file %s\n",file_name);
     *  return -1;
     * }
     *
     * flag=get_word(f,str);
     * if(flag==-1) return -1;
     * num_messages=atoi(str);
     *
     */

    /* Test parameters */
#define GETWORD flag = get_word( f, str , READ_STR_LENGTH ); if ( flag ) return -1;
#define FORMATCHECK(x) if ( strcmp( str, x ) ) { printf( "Bad format file %s\n", file_name ); return -1; }


    GETWORD FORMATCHECK( "test" )
    GETWORD FORMATCHECK( "type" )
    flag = read_string( f, test_type, READ_STR_LENGTH );
    if ( flag )
        return -1;

	GETWORD FORMATCHECK( "data" )
	GETWORD FORMATCHECK( "type" )
	flag = read_string( f, test_type, READ_STR_LENGTH );
	if ( flag )
		return -1;

    GETWORD FORMATCHECK( "begin" )
    GETWORD FORMATCHECK( "message" )
    GETWORD FORMATCHECK( "length" )
    GETWORD begin_message_length = atoi( str );

    GETWORD FORMATCHECK( "end" )
    GETWORD FORMATCHECK( "message" )
    GETWORD FORMATCHECK( "length" )
    GETWORD end_message_length = atoi( str );

    GETWORD FORMATCHECK( "step" )
    GETWORD FORMATCHECK( "length" )
    GETWORD step_length = atoi( str );

    GETWORD FORMATCHECK( "noise" )
    GETWORD FORMATCHECK( "message" )
    GETWORD FORMATCHECK( "length" )
    GETWORD noise_message_length = atoi( str );

    GETWORD FORMATCHECK( "number" )
    GETWORD FORMATCHECK( "of" )
    GETWORD FORMATCHECK( "noise" )
    GETWORD FORMATCHECK( "messages" )
    GETWORD noise_message_num = atoi( str );

    GETWORD FORMATCHECK( "number" )
    GETWORD FORMATCHECK( "of" )
    GETWORD FORMATCHECK( "noise" )
    GETWORD FORMATCHECK( "processes" )
    GETWORD noise_processors = atoi( str );

    GETWORD FORMATCHECK( "number" )
    GETWORD FORMATCHECK( "of" )
    GETWORD FORMATCHECK( "repeates" )
    GETWORD num_repeats = atoi( str );
    
    /*
     * Commented by Alexey Salnikov.
     *
     * I think commented code fragment is abuse for file format.
     */
    
    /*
    GETWORD FORMATCHECK( "result" )
    GETWORD FORMATCHECK( "file" )
    flag = read_string( f, str );
    if ( flag == -1 ) return -1;
    */

    GETWORD FORMATCHECK( "hosts:" )
    
    // Update state
    state = info_state_test_parameters;

    /* Reading host names, each processor - one host name */
    host_names = (char**)malloc(sizeof(char*)*num_processors);
    host_ranks = (int*)malloc(sizeof(int)*num_processors);
    for ( int i = 0; i < num_processors; i++ )
        host_names[i] = (char*)malloc( 256 * sizeof(char));
    for ( int i = 0; i < num_processors; i++ )
    {
        flag = get_word( f, host_names[i] , READ_STR_LENGTH );
        if ( flag ) return -1;
	
	/*
	 * 
	 * Why we need to describe host rank in file?
	 *
	 * The order of hosts in a file sets natural order of hosts in
	 * a MPI-programm.
	 * 
	 */
	/*
	GETWORD FORMATCHECK( "rank" )
	GETWORD host_ranks[i] = atoi( str );
	*/
    }

#undef GETWORD
#undef FORMATCHECK

    int tmp_num_messages = 0;
    for( int i = begin_message_length; i < end_message_length; tmp_num_messages++ )
        i += step_length;
    messages_length=(int *)malloc(sizeof(int)*tmp_num_messages);
    if(messages_length==NULL)
    {
        printf("Network_speed::fread(char *) Out of the memory\n");
        return -1;
    }

    links=new  Matrix [tmp_num_messages];
    if(links==NULL)
    {
        printf("Network_speed::fread(char *) Out of the memory\n");
        return -1;
    }

    num_messages = 0;
    for(i=0;i<tmp_num_messages;i++)
    {
        flag=get_word(f,str,READ_STR_LENGTH);
        if(flag)
            return -2;
        if(strcmp(str,"Message"))
        {
            printf("Incomplete file %s\n",file_name);
            return -2;
        }

        flag=get_word(f,str,READ_STR_LENGTH);
        if(flag)
            return -2;
        if(strcmp(str,"length"))
        {
            printf("Incomplete file %s\n",file_name);
            return -2;
        }

        flag=get_word(f,str,READ_STR_LENGTH);
        if(flag)
            return -2;
        messages_length[i]=atoi(str);

        flag=links[i].fread(f,num_processors,num_processors);
        if(flag)
        {
            printf("Can not read matrix for the %d message length\n",messages_length[i]);
            return -2;
        }

        num_messages++; // So, we got matrix for another one message length
    }

    // Update state
    if ( num_messages == tmp_num_messages )
        state = info_state_partial_matrices;
    else
        state = info_state_all_done;

    // Close file!
    fclose(f);

    return 0;
}
/****************************************************************************/
double Network_speed::translate_time(int from,int to,int length)
{
	int i;
	
	/*
	 * There is not line interpolation scheme.
	 *
	 * We discussed does line interpolation nesessary 
	 * here and decide it will be other scheme.
	 * 
	 */
	
	for(i=0;i<num_messages;i++)
	{
		if(length <= messages_length[i]) break;
	}
	
	if(i==num_messages)
	{
		return links[num_messages-1].element(from,to);
	}
	
	return links[i].element(from,to);
}
/****************************************************************************/

