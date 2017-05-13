#include "Divisive.h"

Divisive::Divisive(int n = 0, int required = 0, ClustData *data = NULL) : dataset(data), estClustNum(required)
{
	if (n == -1)
		N = data->getInfo().proc_num;
	else
		N = n;
}


std::pair <int, int> Divisive::findFarest(int x, int y, Cluster *cl) {
	std::pair <int, int> result = std::pair <int, int>(x, y);
	std::vector <std::pair <int, int> > clData = cl->getElements();
	double maxDist = 0;
	for (int i = 0; i < clData.size(); i++) {
		double d = dataset->getDist(std::pair <int, int>(x, y), clData[i]);
		if (d > maxDist) {
			result = clData[i];
			maxDist = d;
		}
	}
	return result;
}

void Divisive::divideCluster(Cluster *clf)
{
	Cluster *cl1, *cl2;
	cl1 = new Cluster();
	cl2 = new Cluster();
	std::pair <int, int> a1 = findFarest(clf->getSeed().first, clf->getSeed().second, head);
	std::pair <int, int> a2 = findFarest(a1.first, a1.second, head);
	std::vector <std::pair <int, int> > el1, el2;
	std::vector <std::pair <int, int> > elems = clf->getElements();
	
	Cluster* cfather = clf;
	for (int i = 0; i < elems.size(); i++) {
		if (dataset->getDist(a1, elems[i]) < dataset->getDist(a2, elems[i])) {
			el1.push_back(elems[i]);
		}
		else {
			el2.push_back(elems[i]);
		}
	}
	clf->setChilds(std::pair <Cluster*, Cluster*>(cl1, cl2));
	cl1->setChilds(std::pair <Cluster*, Cluster*>(NULL, NULL));
	cl2->setChilds(std::pair <Cluster*, Cluster*>(NULL, NULL));
	cl1->setFather(clf);
	cl2->setFather(clf);
	cl1->setElements(el1);
	cl2->setElements(el2);
	cl1->setSeed(a1);
	cl2->setSeed(a2);
}



void Divisive::clusterise()
{
	srand(time(0));
	int x = rand() % N;
	int y = rand() % N;
	std::vector <std::pair <int, int> > a;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			a.push_back(std::pair<int, int>(i, j));
	head = new Cluster(a, x, y);
	std::vector <Cluster*> res;
	res.push_back(head);
	while (res.size() != estClustNum) {
		divideCluster(res.front());
		res.push_back(res.front()->getChilds().first);
		res.push_back(res.front()->getChilds().second);
		res.erase(res.begin(), res.begin()+1);
	}
	totalClusters = res.size();
	result = res;
	for (int i = 0; i < res.size(); i++)
		res[i]->calcStats(dataset);
}

void Divisive::printData(std::ofstream & out)
{
	int clustnum = 0;
	for (int i = 0; i < result.size(); i++) {
		if (result[i]->isHollow() == false) {
			out << "Cluster#" << ' ' << clustnum << std::endl;
			result[i]->printData(out);
			clustnum++;
		}
	}
}

Divisive::~Divisive()
{
}
