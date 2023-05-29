#include <glad/glad.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>



static const GLfloat p_vertex_buffer_data[] = {
     0.0f, 0.5f,  0.5f, 
     0.0f, 0.5f, -0.5f, 
     0.0f, -0.5f, -0.5f, 
     0.0f, -0.5f,  0.5f, 
     1.0f, 1.5f, 0.0f
};

static const struct Particle {
    glm::vec4 particleColor;

    glm::vec3 pos, velocity;
    float mass;
};



class Particles {
    public:
        Particles() {};
        const char * texturePath;
        
        void initializeParticles(unsigned int numBodies);
        void display();
        void update();
        void destroy();

    private:  

        cudaGraphicsResource_t resource;
        cudaGraphicsResource_t velocity_resource;
        cudaGraphicsResource_t resources[2];

        float3* buffers[2];

        unsigned int numBlocks;
        unsigned int threadsPerBlock;

        GLuint vertexArrayID;
        GLuint particles_vertex_buffer;
        GLuint velocity_buffer;
        
        unsigned int texture;

        void calculateKernelParams(unsigned int numBodies);
        void createParticleBuffers();
        
        void loadTexture();
};