#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

#include <glad/glad.h>
#include "camera.hpp"

#include "obj_renderer.hpp"
#include "particle_renderer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#include "nbody.cuh"


//must set public variable texturePath before calling!
void Particles::initializeParticles(unsigned int numBodies) {
    if (texturePath != NULL) {
   //     std::cout << "here first" << std::endl;
        calculateKernelParams(numBodies);
        createParticleBuffers();
    //    std::cout << "here second" << std::endl;
        loadTexture();

        size_t size;
        CHECK_CUDA(cudaGraphicsMapResources(1, &resources[0], 0));

        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&buffers[0], &size, resources[0]));

        launchInitKernel(numBlocks, threadsPerBlock, buffers[0], type);

        CHECK_CUDA(cudaGraphicsUnmapResources(1, &resources[0], NULL));

    }
    else {
        std::cout << "No texture path provided" << std::endl;
    }
}

void Particles::calculateKernelParams(unsigned int numBodies) {
    //this inefficient little program only runs once and forces the user to use at least 1024 particles
    if (numBodies % 1024 != 0) {
        std::cout << "Input particle count must be divisible by 1024. Adjusting number of particles..." << std::endl;
    }
    while (numBodies % 1024 != 0) {
        numBodies++;
    }

    numBlocks = numBodies / 1024;
    threadsPerBlock = 1024;
}

void Particles::update() {
    
    size_t size;
    
    CHECK_CUDA(cudaGraphicsMapResources(2, resources, 0));
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&buffers[0], &size, resources[0]));
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&buffers[1], &size, resources[1]));

    launchGravityKernel(numBlocks, threadsPerBlock, buffers[0], buffers[1], type);

    CHECK_CUDA(cudaGraphicsUnmapResources(2, resources, NULL));

    glBindBuffer(GL_ARRAY_BUFFER, particles_vertex_buffer);
}

void Particles::createParticleBuffers() {
	glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);

    GLsizei bufferSize = numBlocks * threadsPerBlock * 3 * sizeof(float);

    glGenBuffers(1, &particles_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
	glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    );

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&resources[0], particles_vertex_buffer, cudaGraphicsMapFlagsNone));
    // exit(1);

    glGenBuffers(1, &velocity_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, velocity_buffer);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_DYNAMIC_DRAW);

     glEnableVertexAttribArray(1);
	glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    ); 

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&resources[1], velocity_buffer, cudaGraphicsMapFlagsNone));

    glBindVertexArray(0);

    std::cout << "buffers created" << std::endl;
}

void Particles::destroy() {
	glDeleteVertexArrays(1, &vertexArrayID);
	glDeleteBuffers(1, &particles_vertex_buffer);
	glDeleteBuffers(1, &velocity_buffer);

    cudaGraphicsUnregisterResource(resources[0]);
    cudaGraphicsUnregisterResource(resources[1]);
}

void Particles::display() {
    glBindVertexArray(vertexArrayID);

    glBindBuffer(GL_ARRAY_BUFFER, particles_vertex_buffer);
	glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    ); 

     glBindBuffer(GL_ARRAY_BUFFER, velocity_buffer);
	glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    ); 
    
    glDrawArrays(GL_POINTS, 0, numBlocks * threadsPerBlock);
}

//loads the texture used to display particles
void Particles::loadTexture() {   
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
	
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_PROGRAM_POINT_SIZE);
	
	int width, height, nrChannels;
    unsigned char *data = stbi_load(texturePath, &width, &height, &nrChannels, 0);
    std::cout << nrChannels  << std::endl;

	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    std::cout << "before"<< std::endl;

	if (data) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        GLenum err;
         while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error: " << err << std::endl;
         }
        glGenerateMipmap(GL_TEXTURE_2D);
	}
	else {
		std::cout << "Failed to load texture" << std::endl;
	}
    std::cout << "after"<< std::endl;
	stbi_image_free(data);
    glEnable(GL_DEPTH_TEST);

	// glEnable(GL_BLEND);
	// glBlendFunc(GL_ONE, GL_ONE);

    std::cout << "Texture loaded" << std::endl;

}