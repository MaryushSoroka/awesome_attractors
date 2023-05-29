#ifndef __NBODY_H__
#define __NBODY_H__

#include <cuda_runtime.h>

#define G 6.674e-11

typedef struct vec3d {
    float x;
    float y;
    float z;
};

void CHECK_CUDA(cudaError_t err);

void setDevice();

void launchInitKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions);

void launchGravityKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities);

__global__ void plane(float3* positions);

__global__ void gravityKernel(float3* positions, float3* velocities, float mass, float dt);

__global__ void lorenzKernel(float3* positions, float3* d_velocity, float mass, float dt);

__global__ void pickoverKernel(float3* positions, float3* d_velocity, float mass, float dt);

#endif