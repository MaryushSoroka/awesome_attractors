#include <glad/glad.h>

#include <iostream>
#include <vector>
#include <glm/glm.hpp>

#include <obj_renderer.hpp>

//used for objects loaded using the objloader
void Renderer::draw(std::vector<glm::vec3> vertices) {
    // 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
		glVertexAttribPointer(
			1,                                // attribute
			2,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, vertices.size());
}

//creates buffers for objloader objects
void Renderer::createBuffers(std::vector<glm::vec3> vertices, std::vector<glm::vec2> uvs) {
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);
}

void Renderer::deleteBuffers() {
    glDeleteBuffers(1, &vertexBuffer);
	glDeleteBuffers(1, &uvBuffer);
}