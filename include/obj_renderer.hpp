#include <glad/glad.h>

#include <vector>
#include <glm/glm.hpp>

class Renderer {
	public:
		Renderer() {};
		void draw(std::vector< glm::vec3 > vertices);
		void drawParticles();
		void createBuffers(std::vector< glm::vec3 > vertices, std::vector<glm::vec2> uvs);

		void loadTexture(const char* filePath);
		void bindTexture();

		void createParticleBuffers();
		void deleteBuffers();
		void deleteParticles();

	private:
		GLuint vertexBuffer;
		GLuint uvBuffer;


		GLuint vertexArrayID;
		GLuint particlesVAO;
		GLuint particlesEBO;
		GLuint particles_vertex_buffer;
    	GLuint particles_position_buffer;
   		GLuint particles_color_buffer;

		unsigned int texture;

		static const unsigned int numParticles = 1;
};