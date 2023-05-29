#include <glad/glad.h>


class Shaders {

	public:

		Shaders() {};
		const char* vertex_file_path;
		const char* fragment_file_path;
		const char* compute_file_path;

		GLuint loadShaders();

		GLuint addComputeShader();

};