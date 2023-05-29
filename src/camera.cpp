#include <glad/glad.h>

#include <camera.hpp>

#include <iostream>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//takes the horizontal camera angle, vertical camera angle, camera fov, camera speed and mouse speed
void Camera::updateCamera(GLFWwindow* window) {

    //settings
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    
    static double lastTime = glfwGetTime();
    double currentTime = glfwGetTime();
    deltaTime = float(currentTime - lastTime);

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // we don't have to normalize the horizontal angle because it has infinite scrolling

    double normal_xpos = double (width/2 - xpos ) / double (width/2);
    horizontalAngle = normal_xpos * 1.57;

    //vertical angle should be 0 at (width/2, height/2) and +/- 1.57 at (0, --) and (height, --) (1.57 radians = 90 degrees)
    //so, we can make a function which relates the vertical angle to the ypos, and if we use normalized screen coordinates,
    //this function would be 1.57 * ypos(normalized)
    //we first need to normalize ypos to give the fraction of the window height, then multiply by 1.57.
    //Returns a normal_ypos between -1.0 and 1.0
    double normal_ypos = (double (height/2) - ypos) / double (height/2);
    verticalAngle = normal_ypos * 1.57;

    //keep the camera from flipping over
    if (verticalAngle >= 1.5708) {
        verticalAngle = 1.5708;
        glfwSetCursorPos(window, xpos, 0);
    } 
    if (verticalAngle <= -1.5708) {
        verticalAngle = -1.5708;
        glfwSetCursorPos(window, xpos, height);
    }
    
    direction = glm::vec3(
        cos(verticalAngle) * sin(horizontalAngle),
        sin(verticalAngle),
        cos(verticalAngle) * cos(horizontalAngle)
    );

    right = glm::vec3(
        sin(horizontalAngle - (3.14f / 2.0f)),
        0,
        cos(horizontalAngle - (3.14f / 2.0f))
    );

    up = glm::cross( right, direction );

    // Move forward
    if (glfwGetKey(window, GLFW_KEY_W ) == GLFW_PRESS){
        position += direction * deltaTime * speed;
    }
    // Move backward
    if (glfwGetKey( window, GLFW_KEY_S ) == GLFW_PRESS){
        position -= direction * deltaTime * speed;
    }
    // Strafe right
    if (glfwGetKey(window,  GLFW_KEY_D ) == GLFW_PRESS){
        position += right * deltaTime * speed;
    }
    // Strafe left
    if (glfwGetKey( window, GLFW_KEY_A ) == GLFW_PRESS){
        position -= right * deltaTime * speed;
    }
    if (glfwGetKey( window, GLFW_KEY_E ) == GLFW_PRESS){
        position += up * deltaTime * speed;
    }
    if (glfwGetKey( window, GLFW_KEY_LEFT_SHIFT ) == GLFW_PRESS){
        position -= up * deltaTime * speed;
    }

    std::cout << position.x << " " << position.y << " " << position.z << "      ";
    std::cout << horizontalAngle << " " << verticalAngle << " " << std::endl;

    // Projection matrix : 45&deg; Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    projectionMatrix = glm::perspective(glm::radians(fov), (float) width / (float) height, 1.0f, 10000000.0f);

    // Camera matrix
    viewMatrix = glm::lookAt(position, position+direction, up);
}

//call after updating the camera
glm::mat4 Camera::getMvp() {

    glm::mat4 model = glm::mat4(1.0f);
    return projectionMatrix * viewMatrix * model;
}

cameraProperties Camera::getProperties() {
    properties.cameraRightWS = {viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]};
    properties.cameraUpWS = {viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]};

    return properties;
}