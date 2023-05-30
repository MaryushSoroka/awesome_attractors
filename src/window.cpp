#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include <camera.hpp>
#include <attractor.hpp>
#include <shaders.hpp>
#include <objloader.hpp>
#include <obj_renderer.hpp>
#include <particle_renderer.hpp>

#include "nbody.cuh"

#include <glm/gtc/matrix_transform.hpp>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

//variables
const unsigned int width = 1920;
const unsigned int height = 1080;

GLuint matrixID;
glm::mat4 mvp;

bool pause = false;

int main(int argc, char** argv) 
{
    
    setDevice();

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "astro-sim", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    // glfwSwapInterval(0);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    } 

    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* version = glGetString(GL_VERSION);
    std::cout << vendor << std::endl;
    std::cout << version << std::endl;

    char* p;
    int numBodies;
    //this is horrible but i'm tired of recompiling
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-c") == 0) {
            numBodies = int(strtol(argv[i + 1], &p, 10));
            if (numBodies < 0) {
                std::cerr << "Cannot use negative particle counts" << std::endl;
                numBodies = 1024;
            }
        }
    }

    //initialize class members
    Attractor *attr = new Chen();

    Camera camera;
    camera.position = attr->cam_position;
    camera.speed = attr->cam_speed; 

    Shaders shaders;
    Particles particles;
    
    //Compile vertex and fragment shaders  
    shaders.vertex_file_path = "../shaders/vertex.vert";
    shaders.fragment_file_path = "../shaders/fragment.frag";

    GLuint programID = shaders.loadShaders();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    
    particles.texturePath = "../textures/test_texture.png";
    // particles.type = "Chen";  //Pickover, Chen, Lorenz, Izawa
    particles.initializeParticles(numBodies, attr);
    glEnable(GL_PROGRAM_POINT_SIZE); 


    double lastTime = glfwGetTime();
    int nbFrames = 0;
    double ms = 0;

    // render loop
    while (!glfwWindowShouldClose(window))
    {  
        processInput(window);

        camera.updateCamera(window);
        mvp = camera.getMvp();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glUseProgram(programID);

        glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

        if (!pause)
            ms = particles.update(attr);

        double currentTime = glfwGetTime();
        nbFrames++;
        if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1 sec ago
            // printf and reset timer
            printf("%f ms/frame; FPS: %f\n", 1000.0/double(nbFrames), double(nbFrames));
            std::cout << "CUDA frame calculation time, ms: " << 1000. * ms << std::endl;
            nbFrames = 0;
            lastTime += 1.0;
        }

        particles.display();
        
        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    //delete various things
    particles.destroy();
	glDeleteProgram(programID);

    //end program
	glfwTerminate();
    return 0;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        pause = pause ? false : true;
    }
}

//check if window should close
void processInput(GLFWwindow *window) {
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);     
}

//check if window has been resized
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}