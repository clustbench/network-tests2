#pragma once
#include <vector>
#include <string>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>


using namespace std;


string int_to_str(int a);

class RenderGrid
{
public:
	typedef struct Point {
		vector <glm::vec3> color;
		vector <glm::vec3> pos;
	} Point;
	RenderGrid(int num_proc, int mes_total, vector <int> jumps);
	RenderGrid(int num_proc, int mes_total);
	RenderGrid(hid_t file, int cluster_num, int, int);
	glm::vec2* calc_trans(int& amount);
	vector <GLfloat> mesh();
	int n_elems;
	int num_proc;
	vector <int> jumps;
	~RenderGrid();

private:
	vector <GLfloat> RenderGrid::calc_mesh(int beg, int end, glm::vec4);
	vector <GLfloat> RenderGrid::gen_mesh(GLfloat zbeg, GLfloat zend, glm::vec4 color);
	vector <vector <bool>> toDraw;
	int K;
	int mes_total;
	int mode = 0;
	double one_len_size;
};

