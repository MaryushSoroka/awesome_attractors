#ifndef __NBODY_H__
#define __NBODY_H__
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define G 6.674e-11

typedef struct vec3d {
    float x;
    float y;
    float z;
};

void CHECK_CUDA(cudaError_t err);

void setDevice();

extern "C" void launchInitKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, char* type);

extern "C" void launchGravityKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities, char* type);

extern "C" __global__ void plane(curandState *states, float3* positions, float* bounds);

__global__ void gravityKernel(float3* positions, float3* velocities, float mass, float dt);

extern "C" __global__ void lorenzKernel(float3* positions, float3* d_velocity, float mass, float dt);

//__global__ void pickoverKernel(float3* positions, float3* d_velocity, float mass, float dt);

__global__ void chenKernel(float3* positions, float3* d_velocity, float mass, float dt);

__global__ void IsawaKernel(float3* positions, float3* d_velocity, float mass, float dt);

#endif