#ifndef __CLUST_H__
#define __CLUST_H__

#include <stdlib.h>
#include <set>
#include <vector>

#include "file.h"

#define CLUST_PARAM_PREC_REL		0x00
#define CLUST_PARAM_PREC_ABS		0x01

typedef struct clustering_params_t {
    ///K param in algorithm. 
    size_t first_union_size;
    ///M param in algorithm.
    size_t other_union_size;
    ///Percent of maximum difference between partitions.
    double deviation;
    ///Maximum distance between elements in one cluster.
    double precision;
	///first flag 0x00 - means use precision as relative parameter, 0x01 - as absolute.
	int flags;
} clustering_params_t;

typedef struct index_pair_t {
	unsigned short i, j;
} index_pair_t;

typedef double matrix_value_t;

///Element of matrix which stores also it's coordiantes.
typedef struct matrix_element_t {
	index_pair_t index;
	matrix_value_t value;
} matrix_element_t;

///Index pair comparator.
struct index_pair_less {
	bool operator() (const index_pair_t &A, const index_pair_t &B) const;
};

///Matrix element comporator.
struct matrix_element_less {
	bool operator() (const matrix_element_t &A, const matrix_element_t &B) const;
};

typedef std::set <index_pair_t, index_pair_less> indeces_set_t;
typedef indeces_set_t::iterator indeces_set_iterator_t;

/*typedef indeces_set_t cluster_t;*/

typedef index_pair_t cluster_index_pair_t;
typedef size_t cluster_number_t;
typedef cluster_number_t cluster_t;
typedef std::vector <cluster_t> partition_t;

///Some simple realization of big integers with only two functions: get "digit" and compare two.
template <typename Type>
class LongIndex {
public:
	bool operator <  (const LongIndex <Type>&) const;
	Type operator [] (size_t) const;

	///distance between two vectors as Haming's distance.
	size_t distance (const LongIndex <Type>&) const;

	LongIndex <Type> (): values_ (0) {};

	///@param values - an array to initialize.
	LongIndex <Type> (size_t size, Type *values): values_ (values, (values == NULL)?values:values+size) {};
private:
	typedef std::vector <Type> array_t_;
	typedef typename array_t_::iterator iterator_t_;
	array_t_ values_;
};

///Gets information about file: number of matrces in file, for example.
///@param input - input file (stream or something else).
///@param file_info_structure - structure of header.
///@return - for error processing. 0 means end of file.
int GetFileInfo (FileReader &input, file_info_t *file_info_structure);

/*
///Reads one matrix.
///@param input - input file (stream or something else).
///@param matrix - matrix of values, feeded to Clusterize ().
///@return - for error processing. 0 means end of file.
int ReadMatrix (FileReader &input, matrix_element_t *matrix);*/

/*
///Writes one matrix (possibly in formated way - as partition and mean value for each cluster)
///@param output - output file (stream or something else).
///@param output_format - structure of partition with mean value, representing clusterized matrix.
///@return - for error processing.
int WriteMatrix (FileWriter &output, matrix_element_t *matrix);*/

///Runs clusterisation for file and saves compressed data.
///@param input - input file (stream or something else).
///@param output - output file (stream or something else).
///@param params - preferences for clusterization process.
void RunClustering (FileReader &input, FileWriter &output, clustering_params_t &params);

///Clusterizes one dimensional array.
///@param input - input sorted array.
///@param size - array size.
///@param n - number of elements to pick first time in cluster.
///@param precision - maximal difference between minimal and maximal elements in cluster.
///@param clusters - array of clusters' beginings. Number of clusters is less or equal to input array size.
///@return - real number of clusters.
size_t Clusterize (const matrix_element_t input [], size_t size, size_t n, double precision, size_t clusters []);

///Interesects several partitions into one. Element will stay in cluster of partition if it has less or equal to
///deviation number of errors. So if the most of partitions have concreete elements in one cluster, then this elements 
///will be in the one cluster in the resulting partition.
///@param partitions - array of size n of partitions.
///@param n - number of partitions.
///@param deviation - max distance (hamings) between indeces.
///@param output - output partition.
///@param comparision_flag - additional condition, when processing intersection. If equal to 1 then first partition is more valueble.
size_t DeviantIntersection (const partition_t partitions [], size_t n, size_t deviation, partition_t &output, int comparision_flag = 0);

///Counts the power of decarts multiply of partitions.
///@return The power of decarts multiply of partitions.
size_t PartitionIntersectionSize (const partition_t &A, const partition_t &B);

#endif