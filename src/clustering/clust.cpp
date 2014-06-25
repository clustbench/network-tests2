#include "clust.h"
#include <set>
#include <math.h>
#include <map>
#include <algorithm>

#define EPS__	0.000000000000001

bool index_pair_less::operator() (const index_pair_t &A, const index_pair_t &B) const {
    return (A.i < B.i)||((A.i == B.i)&&(A.j < B.j));
}

bool matrix_element_less::operator() (const matrix_element_t &A, const matrix_element_t &B) const {
    return A.value < B.value;
}

/******************************************************************************/

int GetFileInfo(FileReader &input, file_info_t *file_info_structure) {
    input.GetFileInfo(file_info_structure);
    return 1;
}

/******************************************************************************/

#ifndef __max_
#define __max_(a,b) ((a)>(b))?(a):(b)
#endif

#ifndef __print_arr_
#define __print_arr_(arr,len) for(size_t __xxxx_ = 0; __xxxx_ < (len); ++__xxxx_){printf ("%lu ", (arr)[__xxxx_]);} printf ("\n");
#endif

void Clusters2Partition_(const matrix_element_t matrix [], size_t n_proc, const size_t *clusters, partition_t &partition) {
    size_t length = n_proc*n_proc;
    for (size_t i = 0, j = 0; i < length; ++i) {
        if (i > clusters [j]) {
            ++j;
        }
        partition [matrix [i].index.i*n_proc + matrix [i].index.j] = j;
    }
}

void RunClustering(FileReader &input, FileWriter &output, clustering_params_t &params) {
    matrix_element_t *matrix = NULL;
    matrix_value_t *temp_matrix = NULL;
	size_t *clusters = NULL;
    partition_t *partitions = NULL, first_partition, result;
    size_t line_number = 0, first_partition_power = 0, partition_power = 0, first_line;
    
    file_info_t file_info;
    
    if (params.first_union_size == 0 || params.other_union_size == 0) {
        /*Error*/
        return;
    }
    
    if (!GetFileInfo(input, &file_info)) {
        /*Error*/
        return;
    }
    
    size_t length = file_info.line_length;
    
    if (length == 0) {
        /*Not realy an error.*/
        return;
    }
    
    size_t n_proc = sqrt(length);
    
    if (n_proc*n_proc != length) {
        /*Error*/
        return;
    }
    
    /*Set matrix size to apropriete*/
    matrix = new matrix_element_t [length];
    temp_matrix = new matrix_value_t [length];
    clusters = new size_t [length];
    size_t n_partitions = __max_(params.first_union_size, params.other_union_size+1);
    partitions = new partition_t [n_partitions];
    
    if (matrix      == NULL ||
		temp_matrix == NULL ||
        clusters    == NULL ||
        partitions  == NULL) {
        /*Error*/
        goto End;
    }
    
    for (size_t i = 0; i < n_partitions; ++i) {
        partitions [i].resize (length);
    }
    
    /*While can read*/
    while (line_number < file_info.nlines) {
        first_line = line_number;
		bool is_first_partition = true;
		size_t union_size = params.first_union_size;
		double deviation;
		do {
			//printf ("Enters here\n");
			if (!is_first_partition) {
				partitions [0] = result;
			}
			size_t k;
			for (k = 0; k < union_size && line_number < file_info.nlines; ++k, ++line_number) {
				if (!input.ReadLine(temp_matrix)) {
					/*Error*/
					goto End;
				}
            
				for (size_t i = 0; i < n_proc; ++i) {
					for (size_t j = 0; j < n_proc; ++j) {
						matrix [i*n_proc + j].index.i = i;
						matrix [i*n_proc + j].index.j = j;
						matrix [i*n_proc + j].value = temp_matrix [i*n_proc + j];
					}
				}
            
				std::sort <matrix_element_t*, matrix_element_less> (matrix,
																	matrix+length,
																	matrix_element_less());
			
			
				double precision = 0.0;
				if (params.flags & CLUST_PARAM_PREC_ABS) {
					precision = params.precision;
				} else {
					size_t min_i = 0;
					while (min_i < length && matrix [min_i].value < EPS__) {
						++min_i;
					}
					precision = (matrix [length - 1].value - matrix [min_i].value)*params.precision;
				}
            
				partition_power = Clusterize(matrix,
												   length,              /*Matrix size*/
												   n_proc,              /*Number of processes*/
												   precision,           /*precision*/
												   clusters);
            
				/*Creating partition*/
				Clusters2Partition_(matrix, n_proc, clusters, partitions [k + (is_first_partition)?0:1]);
				//printf ("partition.size (): %u\n", partitions [k + (is_first_partition)?0:1].size ());
			}
        
			size_t index = k + (size_t) ((is_first_partition)?0:1);
			//printf ("%u = %u + %u\n", index, k, (is_first_partition)?0:1);
			//printf ("partitions ptr : %u\n", partitions);
			DeviantIntersection(partitions, index, 1, result, !is_first_partition);
		        //printf ("ok\n");

			if (is_first_partition) {
				first_partition = result;
				first_partition_power = partition_power;
				union_size = params.other_union_size;
				is_first_partition = false;				
				deviation = 0.0;
			} else {
				//DeviantIntersection(partitions, params.other_union_size, 1, result, 1);
				
				size_t new_partition_power = PartitionIntersectionSize (first_partition, result);
				deviation = (double)(new_partition_power - first_partition_power) / first_partition_power;
				//printf ("Almost End\n");
			}
			//printf ("result.size (): %u\n", result.size ());
		} while (deviation < params.deviation && line_number < file_info.nlines);
        
		/*Write result*/

		//printf ("length: %u, result.size(): %u\n", length, result.size ());
        
		if (length != result.size ()) {
			printf ("RunClustering: Unknown error.\n");
			goto End;
		}
		for (size_t i = 0; i < length; ++i) {
			clusters [i] = result [i];
		}
		
		output.WritePartition (line_number, clusters);
    }
    End:
    if (matrix != NULL) {
        delete [] matrix;
    }
    if (temp_matrix != NULL) {
        delete [] temp_matrix;
    }
    if (clusters != NULL) {
        delete [] clusters;
    }
    if (partitions != NULL) {
        delete [] partitions;
    }
}

/******************************************************************************/

size_t Clusterize(const matrix_element_t input [], size_t size, size_t n, double precision, size_t clusters []) {
    size_t i = 0, j = n-1, nclusters = 0;
    
    while (j < size) {
        if (input [j].value - input [i].value < precision) {
            while (j + 1 < size && input [j+1].value - input [i].value < precision) {
                j += 1;
            }
            clusters [nclusters] = j;
            ++nclusters;
            i = j+1;
            j = j+n-1;
        } else {
            if (j < size - 2 && input [j+2].value - input [i+1].value < precision) {
                clusters [nclusters] = i;
                ++nclusters;
                j += 2;
                i += 1;
            } else {
                /*what to do next???*/
                j -= 1;
            }
        }
    }
    
    return nclusters;
}

/******************************************************************************/

typedef LongIndex <size_t> cluster_id_t_;
typedef size_t frequency_t_;
typedef size_t number_t_;
typedef std::map <cluster_id_t_, frequency_t_> cluster_frequency_t_;
typedef std::map <cluster_id_t_, number_t_> cluster_assigment_t_;
typedef cluster_frequency_t_::iterator cluster_frequency_iterator_t_;
typedef cluster_assigment_t_::iterator cluster_assigment_iterator_t_;

size_t DeviantIntersection(const partition_t partitions [], size_t n, size_t deviation, partition_t &output, int comparision_flag) {
    if (n == 0 || partitions == NULL /*|| weights == NULL*/) {
	//printf ("n = %u, partitions = %u\n", n, partitions);
	//printf ("Bad bad error\n");
        return 0;
    }
    
    cluster_frequency_t_ cluster_frequency;
    cluster_assigment_t_ cluster_assigment;
    size_t size = partitions [0].size();
    
    size_t *clusters_vector = new size_t [n];
    size_t number = 0;
    
    /*Counting frequency of different types of cluster intersection*/
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < n; ++j) {
            clusters_vector [j] = partitions [j][i];
        }
        cluster_id_t_ id(n, clusters_vector);
        if (cluster_frequency.find(id) == cluster_frequency.end()) {
            cluster_frequency [id] = 1;
        } else {
            ++cluster_frequency [id];
        }
    }
    
    while (!cluster_frequency.empty()) {
        /*Finding biggest cluster*/
        cluster_frequency_iterator_t_ freq_it, biggest;
        for (freq_it = biggest = cluster_frequency.begin(), ++freq_it; freq_it != cluster_frequency.end(); ++freq_it) {
            if (biggest->second < freq_it->second) {
                biggest = freq_it;
            }
        }
        
        /*Extending biggest cluster and arranging new numbers for clusters*/
        cluster_id_t_ id = biggest->first;
        cluster_assigment [biggest->first] = number;
        cluster_frequency.erase(biggest);
        
        freq_it = cluster_frequency.begin();
        while (freq_it != cluster_frequency.end()) {
            if (id.distance(freq_it->first) <= deviation && (!comparision_flag || id [0] == freq_it->first [0])) {
                cluster_assigment [freq_it->first] = number;
                cluster_frequency.erase(freq_it++);
            } else {
                ++freq_it;
            }
        }
        
        ++number;
    }
    
    /*Creating new partition*/
    output.clear();
    //printf ("size: %u\n", size);
    if (output.size () != size) {
    	output.resize(size);
    }
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < n; ++j) {
            clusters_vector [j] = partitions [j][i];
        }
        cluster_id_t_ id(n, clusters_vector);
        output [i] = cluster_assigment [id];
    }
    
    delete [] clusters_vector;
    return output.size();
}

/******************************************************************************/

typedef std::set <cluster_id_t_> cluster_set_t_;

size_t PartitionIntersectionSize(const partition_t &A, const partition_t &B) {
    if (A.size() != B.size()) {
        return A.size();
    }
    
    cluster_set_t_ cluster_set;
    size_t cluster_vector [2];
    
    for (size_t i = 0; i < A.size(); ++i) {
        cluster_vector [0] = A [i];
        cluster_vector [1] = B [i];
        
        cluster_id_t_ id(2, cluster_vector);
        
        cluster_set.insert(id);
    }
    
    return cluster_set.size();
}

/******************************************************************************/

template <typename Type>
bool LongIndex <Type> :: operator < (const LongIndex <Type> &B) const {
    if (values_.size() > B.values_.size()) {
        return false;
    }
    if (values_.size() < B.values_.size()) {
        return true;
    }
    
    size_t i;
    
    for (i = 0; i < values_.size() && values_ [i] == B.values_ [i]; ++i);
    
    if (i != values_.size() && values_ [i] < B.values_ [i]) {
        return true;
    } else {
        return false;
    }
}

template <typename Type>
Type LongIndex <Type> :: operator [] (size_t i) const {
    if (i < values_.size()) {
        return values_ [i];
    } else {
        return (Type) NULL;
    }
}

template <typename Type>
size_t LongIndex <Type> :: distance(const LongIndex <Type> &B) const {
    size_t distance, size;
    if (values_.size() > B.values_.size()) {
        distance = values_.size() - B.values_.size();
        size = B.values_.size();
    } else {
        distance = B.values_.size() - values_.size();
        size = values_.size();
    }
    
    for (size_t i = 0; i < size; ++i) {
        if (values_ [i] != B.values_ [i]) {
            ++distance;
        }
    }
    return distance;
}

/******************************************************************************/
