#include "ConfRoutine.h"

std::string ConfParser::trim(std::string s) {
	int beg = 0;
	int end = s.size() - 1;
	while (is_delimeter(s[beg])){
		beg++;
	}
	while (is_delimeter(s[end])) {
		end--;
	}
	return s.substr(beg, end - beg + 1);
}

bool ConfParser::is_delimeter(char c) {
	std::string delimeters = " \f\n\r\t\v";
	for (int i = 0; i < delimeters.size(); i++) {
		if (c == delimeters[i])
			return true;
	}
	return false;
}

bool ConfParser::only_delimeters(std::string s) {
	std::string delimeters = " \f\n\r\t\v";
	for (int i = 0; i < s.size(); i++) {
		bool is_delim = false;
		for (int j = 0; j < delimeters.size(); j++)
			if (s[i] == delimeters[j])
				is_delim = true;
		if (is_delim == false)
			return false;
	}

	return true;

}

int ConfParser::parse_file() {
	std::ifstream fs;
	std::string line;
	Section *sect;
	int mode = FileParseMode::SEEK;
	int num_of_line = 0;
	fs.open(filename.c_str());
	while (std::getline(fs, line)) {
		num_of_line++;
		line = trim(line);
		if (mode == FileParseMode::SEEK) {
			if (line[0] == '[' && line[line.size() - 1] == ']') {
				mode = FileParseMode::SECTION;
				sect = new Section();
				sect->name = line.substr(1, line.size() - 2);
				continue;
			} else {
				if (only_delimeters(line)) {
					continue;
				}
				std::cout << "Invalid section declaring: " << num_of_line << ": " << line << std::endl;
				return -1;
			}
		}
		if (mode == FileParseMode::SECTION) {
			if (line.find('=') == std::string::npos) {
				if (only_delimeters(line)) {
					sections.push_back(*sect);
					mode = FileParseMode::SEEK;
					continue;
				} else {
					std::cout << "Invalid parameter declaring(must contain =) " << num_of_line << ": " << line << std::endl;
					return -2;
				}
			}
			sect->params.push_back(line);
		}
	}
	return 0;
}
//5, 4, 1, 3

std::vector <double> ConfParser::parse_args_double(std::string s) {
	int type;
	std::vector <double> result;
	for (int i = 0; i < s.size(); i++) {
		std::string val;
		while(s[i] != ',') {
			if (!is_delimeter(s[i]))
				val.push_back(s[i]);
			i++;
		}
		result.push_back(atof(val.c_str()));
	}
	return result;
}

std::vector <int> ConfParser::parse_args_int(std::string s) {
	int type;
	std::vector <int> result;
	for (int i = 0; i < s.size(); i++) {
		std::string val;
		while(s[i] != ',') {
			if (!is_delimeter(s[i]))
				val.push_back(s[i]);
			i++;
		}
		result.push_back(atoi(val.c_str()));
	}
	return result;
}

int ConfParser::calc_params(ProgParams& r) {
	for (int i = 0; i < sections.size(); i++) {
			if (sections[i].name == "General") {
				for (int j = 0; j < sections[i].params.size(); j++) {
					r.num_proc = -100;
					int ap = sections[i].params[j].find('=');
					std::string param_name = trim(sections[i].params[j].substr(0, ap));
					std::string arg = trim(sections[i].params[j].substr(ap + 1, sections[i].params[j].size() - ap - 1));
					if (param_name == "medians") {
						r.medFile_name = arg;
					} else if (param_name == "deviations") {
						r.devFile_name = arg;
					} else if (param_name == "output") {
						r.outputFile_name = arg;
					} else if (param_name == "num_proc") {
						r.num_proc = atoi(arg.c_str());
					}
					else {
						std::cout << "Wrong parameter name " << param_name << " in section " << sections[i].name << std::endl;
						return -2; 
					}
				}
			} else if (sections[i].name == "DBScan") {
				r.clusteringOptions.push_back(ProgParams::ClusterParams("dbscan"));
				for (int j = 0; j < sections[i].params.size(); j++) {	
					int ap = sections[i].params[j].find('=');
					std::string param_name = trim(sections[i].params[j].substr(0, ap));
					std::string arg = trim(sections[i].params[j].substr(ap + 1, sections[i].params[j].size() - ap - 1));
					if (param_name == "eps_per_experiment") {
						r.clusteringOptions.back().float_parameters = parse_args_double(arg);
					} else if (param_name == "minPts_per_experiment") {
						r.clusteringOptions.back().int_parameters = parse_args_int(arg);
					} else {
						std::cout << "Wrong parameter name " << param_name << " in section " << sections[i].name << std::endl;
						return -2; 
					}	
				}



			} else if (sections[i].name == "Divisive") {
				r.clusteringOptions.push_back(ProgParams::ClusterParams("div"));
				for (int j = 0; j < sections[i].params.size(); j++) {
					int ap = sections[i].params[j].find('=');
					std::string param_name = trim(sections[i].params[j].substr(0, ap));
					std::string arg = trim(sections[i].params[j].substr(ap + 1, sections[i].params[j].size() - ap - 1));
					if (param_name == "clust_number") {
						r.clusteringOptions.back().int_parameters = parse_args_int(arg);
					} else {
						std::cout << "Wrong parameter name " << param_name << " in section " << sections[i].name << std::endl;
						return -2; 
					}
				}

			} else {
				std::cout << "Wrong section name: " << sections[i].name << std::endl;
				return -1;
			}
	}
	return 0;

}