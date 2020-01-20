#include "KMeans.h"



KMeans::KMeans(int K, ClustData clData, double threshold) : K(K), cl_data(clData.getClData()), num_proc(clData.getInfo().proc_num), mes_total((clData.getInfo().end_mes_len - clData.getInfo().begin_mes_len)/clData.getInfo().step_len)
{
	this->threshold = threshold;
	cluster_result.resize(K);
	centroids.resize(K);
	labels.resize(num_proc);
	for (int i = 0; i < num_proc; i++)
		labels[i].resize(num_proc);
}

double KMeans::calc_distance(int m, int n, vector<double> p)
{
	double result = 0.0;
	for (int i = 0; i < p.size(); i++)
	{
		double a = features[m][n][i];
		double b = p[i];
		result += (a - b) * (a - b);
	}
	return result;
}

void KMeans::initialize()
{
	vector <pair<int, int>> rnd;
	srand(time(0));
	for (int i = 0; i < K; i++)
	{
		pair <int, int> a(rand() % num_proc, rand() % num_proc);
		rnd.push_back(a);
		while (check_dissim(rnd))
		{
			a = pair<int, int>(rand() % num_proc, rand() % num_proc);
			rnd[i] = a;
		}
	}
	for (int i = 0; i < rnd.size(); i++)
	{
		centroids[i] = features[rnd[i].first][rnd[i].second];
	}
}

bool KMeans::check_dissim(vector <pair<int, int>> a)
{
	for (int i = 0; i < a.size() - 1; i++)
		for (int j = i + 1; j < a.size(); j++)
	{
			if (a[i] == a[j])
				return true;
	}
	return false;
}

bool KMeans::check_dissim(vector <vector <double>> const &a, vector <vector <double>> const &b)
{
	for (int i = 0; i < a.size(); i++)
		if (a[i] != b[i])
			return true;
	return false;
}

vector < vector <double>> KMeans::recalculate_centroids()
{
	vector < vector <double>> old_centroid = centroids;
	vector <int> calc;
	calc.resize(K);
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < centroids[i].size(); j++)	{
			centroids[i][j] = 0.0;
		}
	}
	for (int i = 0; i < num_proc; i++)
		for (int j = 0; j < num_proc; j++) {
			calc[labels[i][j]]++;
			for (int k = 0; k < mes_total; k++)
			{
				centroids[labels[i][j]][k] += features[i][j][k];
			}
		}
	for (int i = 0; i < K; i++)
		for (int j = 0; j < mes_total; j++)
			centroids[i][j] /= static_cast<double>(calc[i]);
	return old_centroid;
}


void KMeans::clusterise()
{
	calc_features();
	initialize();
	bool changed = true;
	while (changed) {
		changed = false;
		#pragma omp parallel
		{
		#pragma omp for
			for (int i = 0; i < num_proc; i++)
			{
				for (int j = 0; j < num_proc; j++)
				{
					double min_distance = numeric_limits<double>::max();
					for (int k = 0; k < K; k++)
					{
						double curr_distance = calc_distance(i, j, centroids[k]);
						if (min_distance > curr_distance)
						{
							min_distance = curr_distance;
							labels[i][j] = k;
						}
					}
				}
			}
		}
		vector <vector <double>> old = recalculate_centroids();
		changed = check_dissim(old, centroids);
	}
//	cout << "12312";
}

void KMeans::calc_features()
{
	features.resize(num_proc);
	for (int i = 0; i < num_proc; i++) {
		features[i].resize(num_proc);
		for (int j = 0; j < num_proc; j++) {
			features[i][j].resize(mes_total);
		}
	}
	for (int i = 0; i < num_proc; i++)
		for (int j = 0; j < num_proc; j++) {
			median_filter(cl_data[i][j].med);
			for (int k = 0; k < mes_total - 1; k++) {
				features[i][j][k] = cl_data[i][j].med[k + 1] - cl_data[i][j].med[k];
			}
			features[i][j][mes_total - 1] = features[i][j][mes_total - 2];
			features[i][j][0] = features[i][j][1];
		}
	
	for (int i = 0; i < num_proc; i++)
		for (int j = 0; j < num_proc; j++) {
			double sdev = std_dev(features[i][j]);
			double mn = mean(features[i][j]);
			if (sdev <= 0)
				sdev = 1;
			for (int k = 0; k < mes_total; k++) {
				features[i][j][k] = (features[i][j][k] - mn) / (sdev * sqrt(dev_const(mes_total - 1)));
			}
		}
	cout << "Features calculated" << endl;
}

void KMeans::convert(ClustData &raw_vals) {
	for (int i = 0; i < num_proc; i++) 
		for (int j = 0; j < num_proc; j++) {
			cluster_result[labels[i][j]].elements.push_back(pair<int, int>(i, j));
		}

	vector <vector <ClustData::elem>> values = raw_vals.getClData();
	for (int i = 0; i < values.size(); i++) {
		for (int j = 0; j < values.size(); j++) {
			median_filter(values[i][j].med);
		}
	}
	for (int i = 0; i < K; i++) {
		cluster_result[i].centroid.resize(centroids[i].size());
		for (int k = 0; k < cluster_result[i].centroid.size(); k++) {
			double centr = 0.0;
			for (int j = 0; j < cluster_result[i].elements.size(); j++) {
				centr += values[cluster_result[i].elements[j].first][cluster_result[i].elements[j].second].med[k];
			}
			centr /= static_cast<double>(cluster_result[i].elements.size());
			cluster_result[i].centroid[k] = centr;
		}
	}
	for (int i = 0; i < K; i++) {
		vector <double> centr = cluster_result[i].centroid;
		vector <double> cl_feature(centr.size());
		median_filter(centr);
		for (int k = 0; k < centr.size() - 1; k++) {
			cl_feature[k] = centr[k + 1] - centr[k];
		}
		cl_feature[mes_total - 1] = cl_feature[mes_total - 2];
		cl_feature[0] = cl_feature[1];
		double sdev = std_dev(cl_feature);
		double mn = mean(cl_feature);
		if (sdev <= 0)
			sdev = 1;
		for (int k = 0; k < cl_feature.size(); k++) {
			cl_feature[k] = (cl_feature[k] - mn) / (sdev * sqrt(dev_const(mes_total - 1)));
		}
		for (int j = 0; j < cl_feature.size(); j++)
			if (cl_feature[j] >= threshold)
				cluster_result[i].features.push_back(j);
	}
}

void KMeans::printData(hid_t outFile_id) {
	H5LTset_attribute_string(outFile_id, "/", "CLUSTERING_TYPE", "KMEANS");
	H5LTset_attribute_int(outFile_id, "/", "K", &K, 1);
	for (int i = 0; i < cluster_result.size(); i++) {
		std::string clustname = "/CLUSTER_" + int_to_str(i);
		hid_t clustgroupid = H5Gcreate(outFile_id, clustname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		cluster_result[i].printData(clustgroupid);
		H5Gclose(clustgroupid);
	}
}