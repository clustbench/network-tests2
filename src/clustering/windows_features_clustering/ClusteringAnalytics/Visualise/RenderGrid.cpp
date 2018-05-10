#include "RenderGrid.h"

RenderGrid::RenderGrid(int num_proc, int mes_total, vector <int> jumps) : num_proc(num_proc), mes_total(mes_total), jumps(jumps)
{
	toDraw.resize(num_proc);
	for (int i = 0; i < num_proc; i++) {
		toDraw[i].resize(num_proc);
	}
}

RenderGrid::RenderGrid(int num_proc, int mes_total) : num_proc(num_proc), mes_total(mes_total)
{
	toDraw.resize(num_proc);
	for (int i = 0; i < num_proc; i++) {
		toDraw[i].resize(num_proc);
	}
}


RenderGrid::RenderGrid(hid_t file, int cluster_num, int mode, int K) : K(K)
{
	this->mode = mode;
	H5LTget_attribute_int(file, "/", "NUM_PROC", &num_proc);
	toDraw.resize(num_proc);
	for (int i = 0; i < num_proc; i++) {
		toDraw[i].resize(num_proc);
	}
	for (int i = 0; i < num_proc; i++)
		for (int j = 0; j < num_proc; j++)
			toDraw[i][j] = false;
	int bml, eml, sl;
	
	hsize_t dims[2];
	hsize_t dim[1];
	H5LTget_attribute_int(file, "/", "BEG_MES_LEN", &bml);
	H5LTget_attribute_int(file, "/", "END_MES_LEN", &eml);
	H5LTget_attribute_int(file, "/", "STEP", &sl);
	mes_total = (eml - bml) / sl;
	string grname = "/CLUSTER_";
	grname += int_to_str(cluster_num);
	hid_t clGroup_id = H5Gopen(file, grname.c_str(), H5P_DEFAULT);
	H5LTget_dataset_info(clGroup_id, "CLUSTER_ELEMENTS", dims, NULL, NULL);
	H5LTget_dataset_info(clGroup_id, "FEATURES", dim, NULL, NULL);
	int *data = new int[dims[0] * 2];
	int *f = new int[dim[0]];
	H5LTread_dataset_int(clGroup_id, "CLUSTER_ELEMENTS", data);
	H5LTread_dataset_int(clGroup_id, "FEATURES", f);
	n_elems = dims[0];
	for (int i = 0; i < dims[0] * 2; i += 2) {
		toDraw[data[i]][data[i + 1]] = true;
	}
	for (int i = 0; i < dim[0]; i++) {
		jumps.push_back(f[i]);
	}
	delete[] data;
	delete[] f;
}

glm::vec2* RenderGrid::calc_trans(int& amount) {
	vector <glm::vec2> transf;
	int bound = num_proc / 2;
	for (int i = -bound; i < bound + num_proc%2; i++)
		for (int j = -bound; j <bound + num_proc % 2; j++)
			if (toDraw[i + bound][j + bound]) {
				glm::vec2 translation;
				translation.x = (float)i * 2.0;
				translation.y = (float)j * 2.0;
				transf.push_back(translation);
			}
	glm::vec2* result = new glm::vec2[transf.size()];
	for (int i = 0; i < transf.size(); i++)
		result[i] = transf[i];
	amount = transf.size();
	return result;
}



vector <GLfloat> basic_mesh = {
	-1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,//1 2
	-1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f, //2 9
	1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f, //3 16
	-1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f, //2 23
	1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//3 30
	1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//4 37

	-1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//5 44
	-1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//6 51
	1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//7 58
	-1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//6 65
	1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//7 72
	1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//8 79

	1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//4 86
	1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//3 93
	1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//7 100
	1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//4 107
	1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//8 114
	1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//7 121

	-1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//1 128
	-1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//2 135
	-1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//5 142
	-1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//2 149
	-1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//5 156
	-1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//6 163

	-1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//2 170
	1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//4 177
	-1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//6 184
	1.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//4 191
	-1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//6 198
	1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//8 205

	-1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//1 212
	-1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//5 219
	1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//3 226
	-1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//5 233
	1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//3 240
	1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  1.0f,//7 247
};

vector <GLfloat> RenderGrid::gen_mesh(GLfloat zbeg, GLfloat zend, glm::vec4 color) {
	vector <GLfloat> m = basic_mesh;
//	for (int i = 2; i < m.size(); i += 6)
//		cout << m[i] << endl;
//	cout << "----------------" << endl;
	for (int i = 3; i < m.size(); i += 7) {
		m[i] = color.r;
		m[i + 1] = color.g;
		m[i + 2] = color.b;
		m[i + 3] = color.a;
	}
	for (int i = 2; i <= 37; i += 7)
		m[i] = zbeg;
	for (int i = 44; i <= 79; i += 7)
		m[i] = zend;
	m[86] = zbeg;
	m[93] = zbeg;
	m[100] = zend;
	m[107] = zbeg;
	m[114] = zend;
	m[121] = zend;
	m[128] = zbeg;
	m[135] = zbeg;
	m[142] = zend;
	m[149] = zbeg;
	m[156] = zend;
	m[163] = zend;
	m[170] = zbeg;
	m[177] = zbeg;
	m[184] = zend;
	m[191] = zbeg;
	m[198] = zend;
	m[205] = zend;
	m[212] = zbeg;
	m[219] = zend;
	m[226] = zbeg;
	m[233] = zend;
	m[240] = zbeg;
	m[247] = zend;
	return m;
}

vector <GLfloat> RenderGrid::calc_mesh(int beg, int end, glm::vec4 color) {
	GLfloat step = 2.0f / static_cast<GLfloat>(mes_total);
	GLfloat beg_step = 1.0f - static_cast<float> (beg)*step;
	GLfloat end_step = 1.0f - static_cast<float> (end)*step;
	vector <GLfloat> mesh = gen_mesh(beg_step, end_step, color);
	return mesh;
}

vector <GLfloat> RenderGrid::mesh() {
	//GLfloat* result = NULL;
	GLfloat alpha = 1.0 / static_cast<GLfloat> (K) * 3;
	if (alpha > 1.0)
		alpha = 1.0;
	vector <GLfloat> junk;
	int beg = 0;
	if (jumps.empty()) {
		if (!mode)
			return basic_mesh;
		else
			return junk;
	}
	int end = jumps[0];
	vector <GLfloat> mesh1;
	if (!mode) {
		mesh1 = calc_mesh(beg, end, glm::vec4(0.0f, 1.0f, 0.0f, alpha));
		for (int j = 0; j < mesh1.size(); j++)
			junk.push_back(mesh1[j]);
	}
	for (int i = 0; i < jumps.size(); i++) {
		if (i == jumps.size() - 1)
			end = mes_total;
		else
			end = jumps[i + 1];
		beg = jumps[i] + 1;
		vector <GLfloat> mesh2 = calc_mesh(jumps[i], jumps[i] + 1, glm::vec4(1.0f, 0.0f, 0.0f, alpha));
		if (!mode)
			mesh1 = calc_mesh(beg, end, glm::vec4(0.0f, 1.0f, 0.0f, alpha));
		for (int j = 0; j < mesh2.size(); j++)
			junk.push_back(mesh2[j]);
		if (!mode)
			for (int j = 0; j < mesh1.size(); j++)
				junk.push_back(mesh1[j]);
		beg = jumps[i] + 1;
	}
	
//	result = new GLfloat[junk.size()];
//	for (int i = 2; i < junk.size(); i += 6)
//		cout << junk[i] << endl;
	return junk;

}

string int_to_str(int a) {
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

RenderGrid::~RenderGrid()
{
}

