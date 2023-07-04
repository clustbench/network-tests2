#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include <sys/types.h>
#include <dirent.h>

#include "clustbench_types.h"
#include "clustbench_plugin_operations.h"


int clustbench_open_benchmark(const char *path_to_benchmark_code_dir, 
                             const char *benchmark_name, 
                             clustbench_benchmark_pointers_t *pointers)
{
    char *path_to_so;
    char *symbol_name;

    pointers->dynamic_library_handler = NULL;
    pointers->short_description       = NULL;
    pointers->print_help              = NULL;
    pointers->print_parameters        = NULL;
    pointers->parse_parameters        = NULL;
    pointers->define_netcdf_vars      = NULL;
    pointers->put_netcdf_vars         = NULL;
    pointers->test_function           = NULL;

    int path_length = 0;
    path_length += strlen(path_to_benchmark_code_dir);
    path_length += 2*strlen(benchmark_name);
    path_length += 6; /* " path_to_dir/benchmark_name/benchmark_name.so" */

    path_to_so=(char *)malloc(path_length*sizeof(char));
    if(path_to_so == NULL)
    {
        fprintf(stderr,"clustbench_open_benchmark: can't allocate memory for path to library\n");
        return 1;
    }
    
    //snprintf записывает в строку(первый аргумент) форматированные данные
    snprintf(path_to_so, path_length, "%s/%s.so", path_to_benchmark_code_dir,
                                       benchmark_name);

    //void *dlopen(const char *filename, int flag);
    //dlopen загружает динамическую библиотеку, имя которой указано в строке
    //filename, и возвращает прямой указатель на начало динамической библы. 
    pointers->dynamic_library_handler = dlopen(path_to_so,RTLD_LAZY);
    if(pointers->dynamic_library_handler == NULL)
    {
        //dlerror возвращает NULL, если не возникло ошибок с момента 
        //инициализации или его последнего вызова. Если вызывать dlerror() 
        //дважды, то во второй раз рез. выполнения всегда будет равен NULL. 
        fprintf(stderr,"clustbench_open_benchmark: %s\n",dlerror());
        return 1;
    }
    
    //выделение памяти под строку с названием бэнчмарка
    symbol_name = (char *)malloc(strlen(benchmark_name)+512);
    if(symbol_name == NULL)
    {
        fprintf(stderr,"clustbench_open_benchmark: can't allocate memory for name of "
                       "symbol from dynamic library\n");
        return 1;
    }

    /*FIXME
     * Need to fix multiple rewtite of benchmark_name in string.
     */
     
     //sprintf отличается от snprintf тем, что в snprintf есть указание длины
     //буфера строки, а в sprintf - нет
    sprintf(symbol_name,"%s_%s",benchmark_name,"short_description");
    
    //void *dlsym(void *handle, char *symbol);
    //dlsym использует указатель на динамическую библиотеку, возвращаемую 
    //dlopen, и оканчивающееся нулем символьное имя, а затем возвращает 
    //адрес, указывающий, откуда загружается этот символ. Если символ не 
    //найден, то возвращаемым значением dlsym является NULL; тем не менее, 
    //правильным способом проверки dlsym на наличие ошибок является 
    //сохранение в переменной результата выполнения dlerror, а затем    
    //проверка, равно ли это значение NULL.  ЗАЧЕМ ЭТО ВООБЩЕ?
    pointers->short_description = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->short_description == NULL)
    {
        fprintf(stderr,"Can't read symbol '%s' from '%s'\n",symbol_name,path_to_so);
        free(symbol_name);
        free(path_to_so);
        return 1;
    }

    //то же самое для print_help
    sprintf(symbol_name,"%s_%s",benchmark_name,"print_help");
    pointers->print_help = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->print_help == NULL)
    {
        fprintf(stderr,"Can't read symbol '%s' from '%s'\n",symbol_name,path_to_so);
        free(symbol_name);
        free(path_to_so);
        return 1;
    }

    //то же самое для print_parametrs
    sprintf(symbol_name,"%s_%s",benchmark_name,"print_parameters");
    pointers->print_parameters = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->print_parameters == NULL)
    {
        fprintf(stderr,"Can't read symbol '%s' from '%s'\n",symbol_name,path_to_so);
        free(symbol_name);
        free(path_to_so);
        return 1;
    }
   
    //то же самое для parse_parametrs
    sprintf(symbol_name,"%s_%s",benchmark_name,"parse_parameters");
    pointers->parse_parameters = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->parse_parameters == NULL)
    {
        fprintf(stderr,"Can't read symbol '%s' from '%s'\n",symbol_name,path_to_so);
        free(symbol_name);
        free(path_to_so);
        return 1;
    }
    
    //то же самое для define_netcdf_vars
    sprintf(symbol_name,"%s_%s",benchmark_name,"define_netcdf_vars");
    pointers->define_netcdf_vars = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->define_netcdf_vars == NULL)
    {
        fprintf(stderr,"Can't read symbol '%s' from '%s'\n",symbol_name,path_to_so);
        free(symbol_name);
        free(path_to_so);
        return 1;
    }
    
    //то же самое для put_netcdf_vars
    sprintf(symbol_name,"%s_%s",benchmark_name,"put_netcdf_vars");
    pointers->put_netcdf_vars = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->put_netcdf_vars == NULL)
    {
        fprintf(stderr,"Can't read symbol '%s' from '%s'\n",symbol_name,path_to_so);
        free(symbol_name);
        free(path_to_so);
        return 1;
    }
    
    //то же самое для free_parametrs
    sprintf(symbol_name,"%s_%s",benchmark_name,"free_parameters");
    pointers->free_parameters = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->free_parameters == NULL)
    {
        fprintf(stderr,"Can't read symbol '%s' from '%s'\n",symbol_name,path_to_so);
        free(symbol_name);
        free(path_to_so);
        return 1;
    }
    
    //то же самое для test_function
    //почему нет четвертого аргумента в кавычках?
    sprintf(symbol_name,"%s",benchmark_name);
    pointers->test_function = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->test_function == NULL)
    {
        fprintf(stderr,"Can't read symbol '%s' from '%s'\n",symbol_name,path_to_so);
        free(symbol_name);
        free(path_to_so);
        return 1;
    }

    free(symbol_name);
    free(path_to_so);

    return 0;
}


int clustbench_close_benchmark_lib(clustbench_benchmark_pointers_t *pointers)
{
    if(pointers->dynamic_library_handler != NULL)
    {
        dlclose(pointers->dynamic_library_handler);
    }

    pointers->dynamic_library_handler = NULL;
    pointers->short_description       = NULL;
    pointers->print_help              = NULL;
    pointers->print_parameters        = NULL;
    pointers->parse_parameters        = NULL;
    pointers->define_netcdf_vars      = NULL;
    pointers->put_netcdf_vars         = NULL;
    pointers->test_function           = NULL;

    return 0;
}

int clustbench_print_list_of_benchmarks(const char *path_to_benchmarks_code_dir)
{
    DIR *base_dir;
    char **benchmark_names=NULL;
    char *path;

    size_t current_size=0, current_allocated=0;

    size_t len;
    size_t i;
    
    struct dirent *entry;

    len=strlen(path_to_benchmarks_code_dir);

    path=(char *)malloc((len+1024)*sizeof(char));
    if(path == NULL)
    {
        fprintf(stderr,"clustbench_print_list_of_benchmarks: can't allocate memory\n");
        return -1;
    }

    char a = getchar();
    base_dir=opendir(path_to_benchmarks_code_dir);
    if(base_dir == NULL)
    {
        fprintf(stderr,"clustbench_print_list_of_benchmarks: can't open directory '%s': %s\n",
                path_to_benchmarks_code_dir,strerror(errno));
        free(path);
        return -1;
    }    

    while((entry=readdir(base_dir))!=NULL)
    {
        if(!strcmp(entry->d_name,".") || !strcmp(entry->d_name,".."))
        {
            continue;
        }
        
        sprintf(path,"%s/%s",path_to_benchmarks_code_dir,entry->d_name);
        if(access(path,R_OK) == 0)
        {
            if(current_size == current_allocated)
            {
                char **tmp;
                tmp=(char **)realloc(benchmark_names,(current_allocated+256)*sizeof(char *));
                if(tmp == NULL)
                {   
                    /*
                     * There memory leak. And we should free all memory.
                     * But indeed it is no matter because program will be terminated. 
                     */
                    free(path);
                    fprintf(stderr,"clustbench_print_list_of_benchmarks: can't allocate memory\n");
                    return -1;
                }
                current_allocated+=256;
                benchmark_names=tmp;
            }

            benchmark_names[current_size]=(char *)malloc(512*sizeof(char));
            if(benchmark_names[current_size]==NULL)
            {
                fprintf(stderr,"clustbench_print_list_of_benchmarks: can't allocate memory\n");
                return -1;
            }
            strcpy(benchmark_names[current_size],entry->d_name);
            //удаляем.so из названия
            benchmark_names[current_size][strlen(benchmark_names[current_size])-3] = '\0';
            current_size++;
        }        
    }
    
    free(path);
    closedir(base_dir);

    for(i = 0; i < current_size; i++)
    {
        clustbench_benchmark_pointers_t pointers;
        if(clustbench_open_benchmark(path_to_benchmarks_code_dir,benchmark_names[i],&pointers))
        {
            fprintf(stderr,"clustbench_print_list_of_benchmarks: problems read code for benchmark '%s'\n",
                   benchmark_names[i]);
            clustbench_close_benchmark_lib(&pointers);
            continue;

        }
        printf("%s - %s\n",benchmark_names[i],pointers.short_description);
        clustbench_close_benchmark_lib(&pointers);       
    }

    for(i = 0; i < current_size; i++)
    {
        free(benchmark_names[i]);
    }
    free(benchmark_names);

    return 0;

}

