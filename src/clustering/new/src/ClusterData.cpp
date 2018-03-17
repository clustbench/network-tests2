#include "ClusterData.h"

double norm(double a, double b)
{
	if (a == 0 && b == 0)
		return 1;
	else
		return (a * a + b * b);
}

double denorm(double a)
{
	if (a != 0 && fabsf(a) < std::numeric_limits<float>::min()) {
		return 0.0;
	}
	else
	{
		return a;
	}
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
			double *m, *d;
			m = new double[a.sz[0]];
			d = new double[a.sz[0]];
			for (size_t k = 0; k < a.sz[0]; k++) {
				m[k] = datamed[j + i * a.sz[1] + k * a.sz[1] * a.sz[1]];
				d[k] = datadev[j + i * a.sz[1] + k * a.sz[1] * a.sz[1]];
			}
			ins.med = m;
			ins.dev = d;
			clData[i][j] = ins;
		}
	fileinfo = a;
}

double ClustData::getMed(std::pair<int, int> a, int len)
{
	return clData[a.first][a.second].med[len];
}

double ClustData::getDist(std::pair<int, int> a, std::pair<int, int> b)
{
	double n = 0.0;
	double d = 0.0;
	for (int i = 0; i < (fileinfo.end_mes_len - fileinfo.begin_mes_len) / fileinfo.step_len + 1; i++) {
		double p = denorm(clData[a.first][a.second].med[i]), q = denorm(clData[b.first][b.second].med[i]), s = denorm(clData[a.first][a.second].dev[i]), e = denorm(clData[b.first][b.second].dev[i]);
		d += std::abs(p * p - q * q) / norm(s, e);
		n += norm(s, e);
	}
	return d / n;
}

Cluster::Cluster() : childs(std::pair <Cluster*, Cluster*>(NULL, NULL)), father (NULL), seed(std::pair<int, int> (0, 0)) {}

Cluster::Cluster(std::vector <std::pair <int, int> > in, int x, int y) : elements(in), childs(std::pair <Cluster*, Cluster*>(NULL, NULL)), seed(std::pair<int, int>(x, y)) {}

void Cluster::calcStats(ClustData *dataset)
{
	info a = dataset->getInfo();
	m.resize((a.end_mes_len - a.begin_mes_len) / a.step_len + 1);
	d.resize((a.end_mes_len - a.begin_mes_len) / a.step_len + 1);
	for (int i = 0; i < (a.end_mes_len - a.begin_mes_len) / a.step_len + 1; i++) {
		double med = 0, dev = 0;
		for (int j = 0; j < elements.size(); j++) {
			med += denorm(dataset->getMed(elements[j], i));
		}
		med /= elements.size();
		m[i] = med;
		for (int j = 0; j < elements.size(); j++) {
			dev += m[i] - denorm(dataset->getMed(elements[j], i));
		}
		dev /= elements.size();
		d[i] = dev;
	}
}

void Cluster::printData(const hid_t& out_gr)
{
	int* data;
	int sz = elements.size();

	data = new int[sz*2];

	herr_t status;
	hsize_t dims[2];
	dims[0] = sz;
	dims[1] = 2;
	// int sas[10][2];
	// for (int i = 0; i < 10; i++)
	// 	for (int j = 0; j < 2; j++)
	// 		sas[i][j] = 100;
	
	for (int i = 0; i < elements.size(); i++) {
			//data[i*2 + j] = 999;
			//**(data + i * sizeof(int) * 2 + j * sizeof(int)) = 1000;

	
		data[i*2] = elements[i].first;
		//std::cout << elements[i].first << " " << elements[i].second <<std::endl;
		data[i*2 + 1] = elements[i].second; 
	}
	// for (int i = 0; i < sz; i++) {
	// 	for (int j = 0; j < 2; j++)
	// 		std::cout << data[i][j] << ' ';
	// 	std::cout << std::endl;
	// }
	//std::cout << sz << std::endl;
	//std::cout << "123123213213111312313" << std::endl;
	hid_t datsp_id = H5Screate_simple(2, dims, NULL);
	//std::cout << out_gr << std::endl;
	std::string clname = "CLUSTER_ELEMENTS";
	hid_t dat_id = H5Dcreate(out_gr, clname.c_str(), H5T_STD_I32BE, datsp_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(dat_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	std::cout << status << std::endl;
	H5Dclose(dat_id);
	H5Sclose(datsp_id);

	delete data;
	// for (int i = 0; i < elements.size(); i++) {
	// 	out << "(" << elements[i].first << ", " << elements[i].second << ") ";
	// }
	// out << std::endl;
	// for (int i = 0; i < m.size(); i++) {
	// 	out << m[i] << ' ';
	// }
	// out << std::endl;
	// for (int i = 0; i < d.size(); i++) {
	// 	out << d[i] << ' ';
	// }
	// out << std::endl;
}

void Cluster::setChilds(std::pair<Cluster*, Cluster*> s)
{
	childs = s;
}

void Cluster::setFather(Cluster* s)
{
	father = s;
}

void Cluster::setElements(std::vector<std::pair<int, int> > s)
{
	elements = s;
}

void Cluster::setSeed(std::pair<int, int> s)
{
	seed = s;
}

bool Cluster::isHollow()
{
	if (elements.size() == 0)
		return true;
	else
		return false;
}

std::pair<Cluster*, Cluster*> Cluster::getChilds()
{
	return childs;
}

Cluster* Cluster::getFather()
{
	return father;
}

std::vector<std::pair<int, int> > Cluster::getElements()
{
	return elements;
}

std::pair<int, int> Cluster::getSeed()
{
	return seed;
}

std::string int_to_str(int a) {
	std::string result = "";

	std::string inverse = "";
	if (a == 0) {
			
		result = "0";
		//std::cout << "12421414" << std::endl;
		return result;
	}
	while(a > 0) {
		char c = a % 10 + 48;
		inverse.push_back(c);
		a /= 10;
	}
	for (int i = inverse.size() - 1; i >= 0; i--) {
		result.push_back(inverse[i]);

	}
	std::cout << result << std::endl;
	return result;


}