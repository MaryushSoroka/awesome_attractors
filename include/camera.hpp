#include "glm/glm.hpp"
#include <GLFW/glfw3.h>

struct cameraProperties {
    glm::vec3 cameraRightWS;
    glm::vec3 cameraUpWS;
};

class Camera {
    public:
        
        float speed = 0.01f;
        float mouseSpeed = 3.0f;
        float boost_speed = 0.07f;

        Camera() {};

        void updateCamera(GLFWwindow* window);
        glm::mat4 getMvp();
        cameraProperties getProperties();

    private:
        cameraProperties properties;

        glm::mat4 projectionMatrix;
        glm::mat4 viewMatrix;

        glm::vec3 position = glm::vec3(1,0,0);
        float horizontalAngle = 0.0f;
        float verticalAngle = 0.0f;
        float deltaTime;
        float fov = 45.0f;

        glm::vec3 up;
        glm::vec3 right;
        glm::vec3 direction;
};