#include <glm/glm.hpp>
#include <vector>

class Object {
	public:
	
		Object() {};
		const char* filePath;
		void loadObject();
		std::vector <glm::vec3> getVertices();
		std::vector <glm::vec2>  getUvs();
		std::vector <glm::vec3>  getNormals()
		
		;std::vector <glm::vec3> outVertices;
		std::vector <glm::vec2> outUvs;
		std::vector <glm::vec3> outNormals;

	private:
		std::vector <unsigned int> vertexIndices, uvIndices, normalIndices;
		std::vector <glm::vec3> tempVertices;
		std::vector <glm::vec2> tempUvs;
		std::vector <glm::vec3> tempNormals;
		
};