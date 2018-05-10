#include "ClusterData.h"

double norm(double a, double b)
{
	if (a == 0 || b == 0)
		return 1;
	else
		return (1 / (a * a + b * b));
}

void ClustData::fitData(double *datamed, double *datadev, info a)
{
	clData.resize(a.sz[1]);
	for (size_t i = 0; i < a.sz[1]; i++)
		clData[i].resize(a.sz[2]);
	for (size_t i = 0; i < a.sz[1]; i++)
		for (size_t j = 0; j < a.sz[2]; j++)
		{
			elem ins;
			std::vector <double> m(a.sz[0]);
			std::vector <double> d(a.sz[0]);
			for (size_t k = 0; k < a.sz[0]; k++) {
				m[k] = datamed[j + i * a.sz[1] + k * a.sz[1] * a.sz[1]];
				if (datadev != NULL)
					d[k] = datadev[j + i * a.sz[1] + k * a.sz[1] * a.sz[1]];
			}
			ins.med = m;
			ins.dev = d;
			clData[i][j] = ins;
		}
	fileinfo = a;
}

std::vector<std::vector<ClustData::elem>> ClustData::getClData()
{
	return clData;
}

double ClustData::getMed(std::pair<int, int> a, int len)
{
	return clData[a.first][a.second].med[len];
}

std::vector <double> ClustData::get_med_series(std::pair<int, int> a) {
	return clData[a.first][a.second].med;

}

double ClustData::getDist(std::pair<int, int> a, std::pair<int, int> b)
{
	double n = 0.0;
	double d = 0.0;
	for (int i = 0; i < (fileinfo.end_mes_len - fileinfo.begin_mes_len) / fileinfo.step_len + 1; i++) {
		d += (clData[a.first][a.second].med[i] - clData[b.first][b.second].med[i]) * (clData[a.first][a.second].med[i] - clData[b.first][b.second].med[i]) *
			norm(clData[a.first][a.second].dev[i], clData[b.first][b.second].med[i]);
		n += norm(clData[a.first][a.second].dev[i], clData[b.first][b.second].med[i]);
	}
	return d * n;
}

Cluster::Cluster() {}

Cluster::Cluster(std::vector <std::pair <int, int> > in, int x, int y) : elements(in) {}

std::string int_to_str(int a) {
	std::string result = "";
	std::string inverse = "";
	if (a == 0) {
		result = "0";
		return result;
	}
	while (a > 0) {
		char c = a % 10 + 48;
		inverse.push_back(c);
		a /= 10;
	}
	for (int i = inverse.size() - 1; i >= 0; i--) {
		result.push_back(inverse[i]);

	}
	return result;
}

void Cluster::printData(hid_t clustgr_id)
{
	int* data;
	int* ft;
	double* centr;
	int sz = elements.size();
	hsize_t ft_sz[1] = { features.size() };
	hsize_t centr_sz[1] = { centroid.size()};
	ft = new int[features.size()];
	centr = new double[centroid.size()];
	data = new int[sz * 2];
	herr_t status;
	hsize_t dims[2];
	dims[0] = sz;
	dims[1] = 2;
	for (int i = 0; i < elements.size(); i++) {
		data[i * 2] = elements[i].first;
		data[i * 2 + 1] = elements[i].second;
	}
	for (int i = 0; i < features.size(); i++) {
		ft[i] = features[i];
	}
	for (int i = 0; i < centroid.size(); i++) {
		centr[i] = centroid[i];
	}
	hid_t datsp_id = H5Screate_simple(2, dims, NULL);
	hid_t dat_id = H5Dcreate(clustgr_id, "CLUSTER_ELEMENTS", H5T_STD_I32LE, datsp_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(dat_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Dclose(dat_id);
	H5Sclose(datsp_id);

	H5LTmake_dataset_int(clustgr_id, "FEATURES", 1, ft_sz, ft);
	H5LTmake_dataset_double(clustgr_id, "CENTROID", 1, centr_sz, centr);

	delete[] data;
	delete[] ft;
	delete[] centr;
}


bool Cluster::isHollow()
{
	if (elements.size() == 0)
		return true;
	else
		return false;
}