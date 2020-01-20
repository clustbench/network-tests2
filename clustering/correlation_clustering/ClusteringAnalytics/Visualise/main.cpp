#include <glad/glad.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <set>
#include <string>
#include <iomanip>
#include "RenderGrid.h"

void processInput(GLFWwindow *window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

const GLchar* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec4 aColor;\n"
"layout (location = 2) in vec2 aOffset;\n"
"uniform float num_proc;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"out vec4 fColor;\n"
"void main()\n"
"{\n"
"fColor = aColor;\n"
"gl_Position = projection * view * vec4((aPos.x + aOffset.x) /num_proc, (aPos.y + aOffset.y)/num_proc, aPos.z, 1.0);\n"
"}\0";

const GLchar* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec4 fColor;\n"
"void main()\n"
"{\n"
"FragColor = fColor;\n"
"}\n\0";

glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = 1024.0f / 2.0;
float lastY = 768.0 / 2.0;
float fov = 45.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;


int main(int argc, char** argv) {
	int mode = 0;
	string title;
	int K = 1;
	if (argc != 3) {
		cout << "Invalid args\n Usage: <clustering_file.h5> <number_of_cluster> or <clustering_file.h5> features\n";
		return 0;
	}
	int cluster_number = 0;
	string filename = string(argv[1]);
	if (string(argv[2]) == "features") {
		mode = 1;
		title = "Features visualising | ";
	}
	else {
		cluster_number = atoi(argv[2]);
		title = "Cluster visualising | ";
	}
	if (cluster_number < 0) {
		cout << "Invalid args\n Usage: <clustering_file.h5> <number_of_cluster> or <clustering_file.h5> features\n";
		return 0;
	}
	hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	
	vector <RenderGrid> grids;
	if (mode) {
		H5LTget_attribute_int(file, "/", "K", &K);
		for (int i = 0; i < K; i++)
			grids.push_back(RenderGrid(file, i, mode, K));
	}
	else {
		grids.push_back(RenderGrid(file, cluster_number, mode, K));
	}
	set <int> all_features;

	for (int i = 0; i < K; i++) {
		for (int j = 0; j < grids[i].jumps.size(); j++)
			all_features.insert(grids[i].jumps[j]);
	}
	if (mode) {
		title += "Number of clusters: ";
		title += int_to_str(K);
		vector <int> cl(all_features.size());
		for (int i = 0; i < cl.size(); i++)
			cl[i] = 0;
		int u = 0;
		for (set<int>::iterator a = all_features.begin(); a != all_features.end(); a++) {
			for (int i = 0; i < K; i++) {
				for (int j = 0; j < grids[i].jumps.size(); j++)
					if (grids[i].jumps[j] == *a)
						cl[u]++;
			}
			u++;
		}
		cout << "Jump Index" << " | " << "Number of Clusters\n" << "------------------------------\n";
		u = 0;
		for (set<int>::iterator a = all_features.begin(); a != all_features.end(); a++) {
			string m = int_to_str(*a);
			string q = int_to_str(cl[u]);
			cout << m << setw(11 - m.size()) << " | " << q << setw(18 - q.size()) << "\n------------------------------\n";
			u++;
		}
	}
	else {
		title += "Cluster Number: ";
		title += int_to_str(cluster_number);
		title += string(", Elements in cluster: ");
		title += int_to_str(grids[0].n_elems);
	}
	vector <GLfloat*> meshes(K);
	vector <vector <GLfloat>> meshes1(K);
	for (int i = 0; i < K; i++)
		meshes1[i] = grids[i].mesh();

	vector <int> inst(K);
	for (int i = 0; i < K; i++)
		inst[i] = meshes1[i].size() / 7;
	vector <int> amounts(K);
	vector <glm::vec2*> transes(K);
	for (int i = 0; i < K; i++)
		if (inst[i] != 0)
			transes[i] = grids[i].calc_trans(amounts[i]);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(1024, 768, title.c_str(), NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSetCursorPosCallback(window, mouse_callback);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		return -1;
	}
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND); 
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glViewport(0, 0, 1024, 768);

	GLuint vertexShader;
	GLuint fragShader;
	GLuint shaderProgram;
	shaderProgram = glCreateProgram();
	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragShader);
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragShader);
	glLinkProgram(shaderProgram);

	GLuint* instVBO = new GLuint[K];
	glGenBuffers(K, instVBO);
	for (int i = 0; i < K; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, instVBO[i]);
		if (amounts[i] != 0)
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * amounts[i], transes[i], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	GLuint* VBO, *VAO;
	VBO = new GLuint[K];
	VAO = new GLuint[K];
	glGenVertexArrays(K, VAO);
	glGenBuffers(K, VBO);
	for (int i = 0; i < K; i++) {
		glBindVertexArray(VAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[i]);
		if (meshes1[i].size() != 0)
		glBufferData(GL_ARRAY_BUFFER, meshes1[i].size() * sizeof(GLfloat), &meshes1[i][0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));

		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, instVBO[i]);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glVertexAttribDivisor(2, 1);
	}

	while (!glfwWindowShouldClose(window)) {

		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		processInput(window);
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(shaderProgram);
		glm::mat4 projection = glm::perspective(glm::radians(fov), 1024.0f/768.0f, 0.1f, 100.0f);
		GLuint loc = glGetUniformLocation(shaderProgram, "projection");
		glUniformMatrix4fv(loc, 1, GL_FALSE, &projection[0][0]);
		glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
		loc = glGetUniformLocation(shaderProgram, "view");
		glUniformMatrix4fv(loc, 1, GL_FALSE, &view[0][0]);
		loc = glGetUniformLocation(shaderProgram, "num_proc");
		glUniform1f(loc, static_cast<float>(grids[0].num_proc));
		for (int i = 0; i < K; i++) {
			glBindVertexArray(VAO[i]);
			glDrawArraysInstanced(GL_TRIANGLES, 0, inst[i], amounts[i]);
		}	
		glBindVertexArray(0);
		glfwPollEvents();
		glfwSwapBuffers(window);
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragShader);
	glDeleteBuffers(K, VBO);
	glDeleteBuffers(K, instVBO);
	glDeleteVertexArrays(K, VAO);
	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = 2.5 * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.2f; 
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(front);
}
