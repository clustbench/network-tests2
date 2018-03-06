#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

struct ProgParams
{
	struct ClusterParams {
		std::vector <double> float_parameters;
		std::vector <int> int_parameters;
		int metric = 0;
		std::string name;
		ClusterParams(std::string name) : name(name) {}
	};
	int num_proc;
	std::vector <ClusterParams> clusteringOptions;
	std::string medFile_name;
	std::string devFile_name;
	std::string outputFile_name;
};

class ConfParser {
private:
	enum FileParseMode {SEEK, SECTION};
	enum ArgType {DOUBLE_VALUE, INTEGER_VALUE};



	struct Section {
		std::string name;
		std::vector <std::string> params;
	};

	std::vector <double> parse_args_double(std::string);
	std::vector <int> parse_args_int(std::string);

	bool is_delimeter(char);
	bool only_delimeters(std::string);
	std::string trim(std::string);
	std::string filename;
	std::vector <Section> sections;
public:
	int parse_file();
	int calc_params(ProgParams&);
	ConfParser (std::string filename) : filename(filename){
		sections = std::vector <Section> ();
	}


};