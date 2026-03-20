#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
    int comm_size = 4;
    int begin_message_length = 10;
    int end_message_length = 50;
    int step_length = 10;
    int num_repeats = 3;
    int amount_of_lengths = (end_message_length - begin_message_length)/step_length;
    int amount_of_msgs_for_one_process_pair = amount_of_lengths*num_repeats;
    int length_arr[amount_of_msgs_for_one_process_pair];
    int cur_step = 0;
    for
        (
            int tmp_mes_size=begin_message_length;
            tmp_mes_size<end_message_length; 
            tmp_mes_size+=step_length
        )
    {
        for (int i = 0; i < num_repeats; i++) {
          length_arr[cur_step*num_repeats + i] = tmp_mes_size;
        }
        cur_step++;
    }
    int comm_size_sq = comm_size*comm_size;
    int my_pid = getpid();
    int **permutations_arr = (int**)malloc(comm_size_sq*sizeof(int*));
    for (int i = 0; i < comm_size_sq; i++) {
        permutations_arr[i] = (int*)malloc(amount_of_msgs_for_one_process_pair*sizeof(int));
    }
    int *cur_permutation;
    for (int i = 0; i < comm_size_sq; i++) {
        cur_permutation = (int*)malloc(amount_of_msgs_for_one_process_pair*sizeof(int));
        cur_permutation[0] = 0;
        srand(my_pid + i);
        for (int j = 0; j < amount_of_msgs_for_one_process_pair; j++) {
            int cur_ind = rand()%(j+1);
            cur_permutation[j] = cur_permutation[cur_ind];
            cur_permutation[cur_ind] = length_arr[j];
        }
        for (int j = 0; j < amount_of_msgs_for_one_process_pair; j++) {
            permutations_arr[i][j] = cur_permutation[j];
        }
        free(cur_permutation);
    }

    for (int i = 0; i < comm_size_sq; i++) {
        printf("process №%d will send msgs to process №%d in following order: ", i/comm_size, i%comm_size);
        for (int j = 0; j < amount_of_msgs_for_one_process_pair; j++) {
            printf("%d, ", permutations_arr[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < comm_size_sq; i++) {
        free(permutations_arr[i]);
    }
    free(permutations_arr);
}