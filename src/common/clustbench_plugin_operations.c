#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <mpi.h>
#include <time.h>

#include <sys/types.h>
#include <dirent.h>

#include "clustbench_types.h"
#include "clustbench_plugin_operations.h"

#define TIME_TO_WAIT 2
#define TIME_DIFF_COEFF 10
#define NSECS_PER_SEC 999999999
#define MEASUREMENTS_AMOUNT 100

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
    pointers->test_function_mashed    = NULL;

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

    sprintf(symbol_name, "%s_%s", benchmark_name, "mash");
    printf("PRINTING SYMBOL_NAME FOR MASHED FUNCTION: %s\n", symbol_name);
    pointers->test_function_mashed = dlsym(pointers->dynamic_library_handler, symbol_name);
    if(pointers->test_function_mashed == NULL)
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
    pointers->test_function_mashed    = NULL;

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

int compare( const void *arg1, const void *arg2 )
{
   return (*(long*)arg1 > *(long*)arg2);
}

long measure_time_diff(int rank, int another_rank, int mode) {
  //printf("ALOHA MEASURING: %d, %d, %d\n",  rank, another_rank, mode);
  // Содержание не важно.
  int data = 1;
  long res;
  MPI_Status status;
  if (mode == 1) {
    // Разогрев
    MPI_Send(&data, 1, MPI_INT, another_rank, 1, MPI_COMM_WORLD);
    MPI_Recv(&data, 1, MPI_INT, another_rank, 1, MPI_COMM_WORLD, &status);
    // Начало измерений: теперь процесс получатель точно ответит СРАЗУ.
    struct timespec beg_time;
    struct timespec end_time;
    struct timespec beg_times[MEASUREMENTS_AMOUNT];
    struct timespec end_times[MEASUREMENTS_AMOUNT];
    for (int i = 0; i < MEASUREMENTS_AMOUNT; i++) {
      clock_gettime(CLOCK_REALTIME, &beg_times[i]);
      MPI_Send(&data, 1, MPI_INT, another_rank, 1, MPI_COMM_WORLD);
      MPI_Recv(&data, 1, MPI_INT, another_rank, 1, MPI_COMM_WORLD, &status);
      clock_gettime(CLOCK_REALTIME, &end_times[i]);
    }
    long another_rank_times[2*MEASUREMENTS_AMOUNT];
    long another_rank_diffs[MEASUREMENTS_AMOUNT];
    MPI_Recv(another_rank_times, 2*MEASUREMENTS_AMOUNT, MPI_LONG, another_rank, 2, MPI_COMM_WORLD, &status);
    for (int i = 0; i < MEASUREMENTS_AMOUNT; i++) {
      //printf("RANKS: %d, %d; INDEX %d: %d, %d, %d\n", rank, another_rank, i, beg_times[i], end_times[i], another_rank_diffs[i]);
      long real_diff = (another_rank_times[2*i]-beg_times[i].tv_sec)*NSECS_PER_SEC+(another_rank_times[2*i+1]-beg_times[i].tv_nsec);
      long awaited_diff = ((end_times[i].tv_sec-beg_times[i].tv_sec)*NSECS_PER_SEC+(end_times[i].tv_nsec-beg_times[i].tv_nsec))/2;
      //printf("RANKS: %d, %d; INDEX %d: %ld, %ld\n", rank, another_rank, i, real_diff, awaited_diff);
      another_rank_diffs[i] = awaited_diff-real_diff;
    }
    qsort(another_rank_diffs, MEASUREMENTS_AMOUNT, sizeof(int), compare);
    res = another_rank_diffs[MEASUREMENTS_AMOUNT/2];
    printf("MEASURED TIME DIFF: FROM %d TO %d EQUALS %ld NANOSECONDS\n", rank, another_rank, res);
  }
  else {
    MPI_Recv(&data, 1, MPI_INT, another_rank, 1, MPI_COMM_WORLD, &status);
    MPI_Send(&data, 1, MPI_INT, another_rank, 1, MPI_COMM_WORLD);
    struct timespec my_time;
    struct timespec my_times[MEASUREMENTS_AMOUNT];
    for (int i = 0; i < MEASUREMENTS_AMOUNT; i++) {
      MPI_Recv(&data, 1, MPI_INT, another_rank, 1, MPI_COMM_WORLD, &status);
      clock_gettime(CLOCK_REALTIME, &my_times[i]);
      MPI_Send(&data, 1, MPI_INT, another_rank, 1, MPI_COMM_WORLD);
    }
    long times_to_send[2*MEASUREMENTS_AMOUNT];
    for (int i = 0; i < MEASUREMENTS_AMOUNT; i++) {
      times_to_send[i*2] = my_times[i].tv_sec;
      times_to_send[i*2+1] = my_times[i].tv_nsec;
    }
    MPI_Send(times_to_send, 2*MEASUREMENTS_AMOUNT, MPI_LONG, another_rank, 2, MPI_COMM_WORLD);
    res = -1;
  }
  return res;
}

int calculate_offsets(int rank, int commSize) {
  int counter = 1;
  int rounds_to_participate = 0;
  MPI_Status status;
  while (counter <= commSize) {
    if (rank % counter == 0) {
      rounds_to_participate += 1;
    }
    counter *= 2;
  }
  counter /= 2;
  if (rank == 0) {
    rounds_to_participate -= 1;
  }
  printf("STARTING PART 1, %d %d %d\n", counter, rounds_to_participate, rank);
  // part 1
  long *time_diffs = NULL;
  if (rank < counter) {
    if (rounds_to_participate > 1) {
      if (rank == 0) {
        time_diffs = (long*)malloc((commSize-1)*sizeof(long));
      }
      else {
        time_diffs = (long*)malloc((rounds_to_participate-1)*sizeof(long));
      }
    }
    int rank_modifier = 1;
    int another_rank;
    
    for (int i = 1; i < rounds_to_participate; i++) {
      another_rank = rank + rank_modifier;
      if (rank == 0) {
        time_diffs[rank_modifier-1] = measure_time_diff(rank, another_rank, 1);
      }
      else {
        time_diffs[i-1] = measure_time_diff(rank, another_rank, 1);
      }
      rank_modifier*=2;
    }
    if (rank != 0) {
      another_rank = rank-rank_modifier;
      measure_time_diff(rank, another_rank, 2);
      if (time_diffs != NULL) {
        int data;
        MPI_Recv(&data, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
        MPI_Send(time_diffs, rounds_to_participate-1, MPI_LONG, 0, 3, MPI_COMM_WORLD);
      }
    }
    else {
      time_diffs[rank_modifier-1] = measure_time_diff(rank, rank_modifier, 1);
      long *buf = (long*)malloc(rounds_to_participate*sizeof(long));
      int cur_step = counter;
      for(int i = rounds_to_participate-1; i > 0; i--) {
        cur_step /= 2;
        for (int j = cur_step; j < counter; j+=cur_step*2) {
          MPI_Send(&i, 1, MPI_INT, j, 3, MPI_COMM_WORLD);
          MPI_Recv(buf, i, MPI_LONG, j, 3, MPI_COMM_WORLD, &status);
          int tmp_step = 1;
          for (int k = 0; k < i; k++) {
            time_diffs[j+tmp_step-1] = buf[k] + time_diffs[j-1];
            tmp_step*=2;
          }
        }
      }
      free(buf);
    }
  //part 2
  }
  else {
    measure_time_diff(rank, rank-counter, 2);
  }

  long part_2_res;
  for (int i = 0; i < commSize-counter; i++) {
    if (rank == i) {
      part_2_res = measure_time_diff(rank, rank+counter, 1);
      long data = -1;
      if (rank != 0) {
        MPI_Recv(&data, 1, MPI_LONG, 0, 5, MPI_COMM_WORLD, &status);
        MPI_Send(&part_2_res, 1, MPI_LONG, 0, 5, MPI_COMM_WORLD);
      }
      else {
        time_diffs[counter-1] = part_2_res;
        for (int another_proc = 1; another_proc < commSize-counter; another_proc++) {
          MPI_Send(&data, 1, MPI_LONG, another_proc, 5, MPI_COMM_WORLD);
          MPI_Recv(&data, 1, MPI_LONG, another_proc, 5, MPI_COMM_WORLD, &status);
          time_diffs[counter+another_proc-1] = data + time_diffs[another_proc-1];
        }
      }
    }
  }
  long my_offset = 0;
  if (rank == 0) {
    for (int another_proc = 1; another_proc < commSize; another_proc++) {
      long cur_offset = time_diffs[another_proc-1];
      MPI_Send(&cur_offset, 1, MPI_LONG, another_proc, 6, MPI_COMM_WORLD);
      if (cur_offset > my_offset) {
        my_offset = cur_offset;
      }
    }
  }
  else {
    MPI_Recv(&my_offset, 1, MPI_LONG, 0, 6, MPI_COMM_WORLD, &status);
    printf("PROCESS WITH RANK %d has following OFFSET: %ld\n", rank, my_offset);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (time_diffs != NULL) {
    free(time_diffs);
  }
  return my_offset;
}

void sync_time(int rank, int commSize, long offset) {
  //Определение разрешающей способности таймера
  struct timespec resolution;
  clock_getres(CLOCK_REALTIME, &resolution);
  long ns_resolution = resolution.tv_nsec; 
  struct timespec my_time;
  clock_gettime(CLOCK_REALTIME, &my_time);
  long time_arr[3];
  if (rank == 0){
    time_arr[0] = my_time.tv_sec;
    time_arr[1] = my_time.tv_nsec;
    // Надбавка ко времени ожидания в зависимости от максимального найденного рассинхрона.
    time_arr[2] = offset*TIME_DIFF_COEFF/NSECS_PER_SEC+1;
  }
  MPI_Bcast(time_arr, 3, MPI_LONG, 0, MPI_COMM_WORLD);
  //fprintf(log_file, "MY RANK: %d. GOT TIME: sec: %ld nsec: %ld\n", rank, time_arr[0], time_arr[1]);
  long wait_until_time = time_arr[0] + TIME_TO_WAIT + time_arr[2];
  struct timespec time_to_sleep;
  long time_to_sleep_nsecs = -offset;
  clock_gettime(CLOCK_REALTIME, &my_time);
  time_to_sleep_nsecs += (wait_until_time - my_time.tv_sec)*NSECS_PER_SEC + (NSECS_PER_SEC - my_time.tv_nsec);
  time_to_sleep.tv_sec = time_to_sleep_nsecs/NSECS_PER_SEC;
  time_to_sleep.tv_nsec = time_to_sleep_nsecs%NSECS_PER_SEC;
  time_to_sleep.tv_nsec -= time_to_sleep.tv_nsec%ns_resolution;
  nanosleep(&time_to_sleep, NULL);
  clock_gettime(CLOCK_REALTIME, &my_time);
}



