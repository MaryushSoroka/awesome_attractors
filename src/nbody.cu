#include <iostream>
#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "particle_renderer.hpp"
#include "nbody.cuh"
#include <curand_kernel.h>

#define DIM 512
#define G 6.674e-11


void CHECK_CUDA(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    }
}

void setDevice() {
    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);

    cudaGLSetGLDevice(dev);

    std::cout << "Using device " << dev << std::endl;
}

extern "C" void launchInitKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, char * type) {
    cudaError_t cudaStatus;

    unsigned int sqrtThreads = (unsigned int)sqrt(threadsPerBlock);
    static curandState *states = NULL;
    cudaGetErrorString(cudaMalloc((void **)&states, sizeof(curandState) *
                sqrtThreads * sqrtThreads*numBlocks));
    // unsigned int sqrtThreads = threadsPerBlock;
    dim3 threads(sqrtThreads, sqrtThreads);
    std::cout<< type<< std::endl;
    float * bounds, *bounds_h;
    cudaMalloc(&bounds, sizeof(float) *6);
    bounds_h = (float *) malloc(sizeof(float)* 6);
    if (strcmp(type, "Lorenz")==0){
        bounds_h[0] = -25.; bounds_h[1] = 20.; bounds_h[2] = 5.; bounds_h[3] = 40.; bounds_h[4] = 10.; bounds_h[5] = 40.;
        cudaMemcpy(bounds, bounds_h, sizeof(float)*6, cudaMemcpyHostToDevice);
    }
    if (strcmp(type, "Pickover")==0){
        bounds_h[0] = -25.; bounds_h[1] = 20.; bounds_h[2] = 5.; bounds_h[3] = 40.; bounds_h[4] = 10.; bounds_h[5] = 40.;
        cudaMemcpy(bounds, bounds_h, sizeof(float)*6, cudaMemcpyHostToDevice);

    }
    if (strcmp(type, "Chen")==0){
        bounds_h[0] = -15.; bounds_h[1] = 15.; bounds_h[2] = -15.; bounds_h[3] = 15.; bounds_h[4] = 10.; bounds_h[5] = 40.;
        cudaMemcpy(bounds, bounds_h, sizeof(float)*6, cudaMemcpyHostToDevice);
    }
    if (strcmp(type, "Izawa")==0){
        bounds_h[0] = -2.; bounds_h[1] =2.; bounds_h[2] = -2.; bounds_h[3] = 2.; bounds_h[4] = -2.; bounds_h[5] = 0.;
        cudaMemcpy(bounds, bounds_h, sizeof(float)*6, cudaMemcpyHostToDevice);
    }
    std::cout<< bounds_h[5]<< std::endl;
    plane<<<numBlocks, threads>>>(states, positions, bounds);
    cudaStatus = cudaGetLastError();


    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error launching initialization kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

void launchGravityKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities) {
    cudaError_t cudaStatus;
    unsigned int sqrtThreads = (unsigned int)sqrt(threadsPerBlock);
    // unsigned int sqrtThreads = threadsPerBlock;
    dim3 threads(sqrtThreads, sqrtThreads);

    float mass = 100000.0; //kg
    // float dt = 100.0; //seconds
    float dt = 0.005;//0.001; //seconds
    if (strcmp(type, "Lorenz")==0){
    lorenzKernel<<<numBlocks, threads>>>(positions, velocities, mass, dt);
    }
    else if (strcmp(type, "Pickover")==0){
  //      pickoverKernel<<<numBlocks, threads>>>(positions, velocities, mass, dt);
    }
    else if(strcmp(type, "Chen")==0){
        chenKernel<<<numBlocks, threads>>>(positions, velocities, mass, dt);
    }
    else if(strcmp(type, "Izawa")==0){
        IsawaKernel<<<numBlocks, threads>>>(positions, velocities, mass, dt);
    }
    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error launching kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

extern "C" __global__ void plane(curandState *states, float3* positions, float * bounds) {
    // unsigned int id = threadIdx.x + threadIdx.y * blockDim.x;
    int tid = threadIdx.x;
    int col_offset = blockDim.x * blockDim.y * blockIdx.x;
    int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
    int id = tid + col_offset + row_offset;
    curandState *state = states + tid;
    curand_init(9384, id, 0, state);

    // curandState *d_state;
    // cudaMalloc(&d_state, sizeof(curandState));
    // unsigned *x_result, *h_result;
    // unsigned *d_max_rand_int, *h_max_rand_int, *d_min_rand_int, *h_min_rand_int;
    // cudaMalloc(&d_result, (MAX-MIN+1) * sizeof(unsigned));
    // h_result = (unsigned *)malloc((MAX-MIN+1)*sizeof(unsigned));
    // cudaMalloc(&d_max_rand_int, sizeof(unsigned));
    // h_max_rand_int = (unsigned *)malloc(sizeof(unsigned));
    // cudaMalloc(&d_min_rand_int, sizeof(unsigned));
    // h_min_rand_int = (unsigned *)malloc(sizeof(unsigned));
    // cudaMemset(d_result, 0, (MAX-MIN+1)*sizeof(unsigned));
    // setup_kernel<<<1,1>>>(d_state);

    // *h_max_rand_int = MAX;
    // *h_min_rand_int = MIN;
    // cudaMemcpy(d_max_rand_int, h_max_rand_int, sizeof(unsigned), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_min_rand_int, h_min_rand_int, sizeof(unsigned), cudaMemcpyHostToDevice);
    // generate_kernel<<<1,1>>>(d_state, ITER, d_max_rand_int, d_min_rand_int, d_result);
    // cudaMemcpy(h_result, d_result, (MAX-MIN+1) * sizeof(unsigned), cudaMemcpyDeviceToHost);
    // printf("Bin:    Count: \n");
    // for (int i = MIN; i <= MAX; i++)
    //     printf("%d    %d\n", i, h_result[i-MIN]);
 
    // positions[id].x = (threadIdx.x) * 50.0;
    // positions[id].y = (blockIdx.x * 50.0) - 1000.0 ;
    // positions[id].z = (threadIdx.y) * 50.0;
   
        float rand = curand_uniform(state);
        positions[id].x = rand*(bounds[1]-bounds[0])+bounds[0];
        rand = curand_uniform(state);
        positions[id].y = rand*(bounds[3]-bounds[2])+bounds[2];
        rand = curand_uniform(state);
        positions[id].z = rand*(bounds[5]-bounds[4])+bounds[4];



    // positions[id].x = 10 * threadIdx.x / 32 - 5;
    // positions[id].y = 10.* threadIdx.y / 32 - 5;
    // positions[id].z = 10. * blockIdx.x / 4 - 5;
}

__global__ void gravityKernel(float3* positions, float3* d_velocity, float mass, float dt) {
    int tid = threadIdx.x;
    int col_offset = blockDim.x * blockDim.y * blockIdx.x;
    int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
    int i = tid + col_offset + row_offset;
    const float3 d0_i = positions[i];
    float3 a = {0, 0, 0};

    for (int j = 0; j < blockDim.x * blockDim.y * gridDim.x; j++) {
        if (j == i) continue;

        const float3 d0_j = positions[j];
        float3 r_ij;
        r_ij.x = d0_i.x - d0_j.x;
        r_ij.y = d0_i.y - d0_j.y;
        r_ij.z = d0_i.z - d0_j.z;

        float r_squared = (r_ij.x * r_ij.x) + (r_ij.y * r_ij.y) + (r_ij.z * r_ij.z);

        float F_coef = -G * mass / r_squared;

        a.x += F_coef * r_ij.x * rsqrt(r_squared);
        a.y += F_coef * r_ij.y * rsqrt(r_squared);
        a.z += F_coef * r_ij.z * rsqrt(r_squared);

       
    }   
        const float3 v0_i = d_velocity[i];
        d_velocity[i].x = v0_i.x + (a.x * dt);
        d_velocity[i].y = v0_i.y + (a.y * dt);
        d_velocity[i].z = v0_i.z + (a.z * dt);

        // positions[i].x = (d0_i.x + v0_i.x * dt + a.x * dt * dt / 2.0);
        // positions[i].y = (d0_i.y + v0_i.y * dt + a.y * dt * dt / 2.0);
        // positions[i].z = (d0_i.z + v0_i.z * dt + a.z * dt * dt / 2.0);
}

extern "C" __global__ void lorenzKernel(float3* positions, float3* d_velocity, float mass, float dt) {
    int tid = threadIdx.x;
    int col_offset = blockDim.x * blockDim.y * blockIdx.x;
    int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
    int i = tid + col_offset + row_offset;
    const float3 d0_i = positions[i];
    float3 a = {0, 0, 0};

    const float sigma = 10.;
    const float rho = 28.;
    const float beta = 8./3.;

    const float3 v0_i = d_velocity[i];
    
    d_velocity[i].x = sigma * (d0_i.y - d0_i.x);
    d_velocity[i].y = d0_i.x * (rho - d0_i.z) - d0_i.y;
    d_velocity[i].z = d0_i.x * d0_i.y - beta * d0_i.z;

    positions[i].x = (d0_i.x + d_velocity[i].x * dt);
    positions[i].y = (d0_i.y + d_velocity[i].y * dt);
    positions[i].z = (d0_i.z + d_velocity[i].z * dt);
}


// __global__ void pickoverKernel(float3* positions, float3* d_velocity, float mass, float dt) {
//     int tid = threadIdx.x;
//     int col_offset = blockDim.x * blockDim.y * blockIdx.x;
//     int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
//     int i = tid + col_offset + row_offset;

//     // parameters 1, 1.8, 0.71, 1.51
//     auto a = 1.;
//     auto b = 1.8; 
//     auto c = 0.71;
//     auto d = 1.51;

//     const float3 d0_i = positions[i];
//     const float3 v0_i = d_velocity[i];



//     float new_x = sin(a * d0_i.y) - d0_i.z * cos(b * d0_i.x);
//     float new_y = d0_i.z * sin(c * d0_i.x) - cos(d * d0_i.y);
//     float new_z = sin(d0_i.x);
//     d_velocity[i].x = (new_x-d0_i.x)*300;
//     d_velocity[i].y = (new_y-d0_i.y)*300;
//     d_velocity[i].z = (new_z-d0_i.z)*300;

//     positions[i].x = new_x;
//     positions[i].y = new_y;
//     positions[i].z = new_z;
//}


__global__ void chenKernel(float3* positions, float3* d_velocity, float mass, float dt) {
    int tid = threadIdx.x;
    int col_offset = blockDim.x * blockDim.y * blockIdx.x;
    int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
    int i = tid + col_offset + row_offset;
    const float3 d0_i = positions[i];

    const float sigma = -0.38;
    const float alpha = 5.;
    const float beta = -10.;

    const float3 v0_i = d_velocity[i];
    d_velocity[i].x = alpha *  d0_i.x- d0_i.y*d0_i.z;
    d_velocity[i].y = beta*d0_i.y+d0_i.x*d0_i.z;
    d_velocity[i].z = sigma*d0_i.z+d0_i.x*d0_i.y/3.;

    positions[i].x = (d0_i.x + d_velocity[i].x * dt);
    positions[i].y = (d0_i.y + d_velocity[i].y * dt);
    positions[i].z = (d0_i.z + d_velocity[i].z * dt);
}

__global__ void IsawaKernel(float3* positions, float3* d_velocity, float mass, float dt) {
    int tid = threadIdx.x;
    int col_offset = blockDim.x * blockDim.y * blockIdx.x;
    int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
    int i = tid + col_offset + row_offset;
    const float3 d0_i = positions[i];

    const float a = 0.95;
    const float b = 0.7;
    const float c = 0.6;
    const float d = 3.5;
    const float e = 0.25;
    const float f = 0.1;

    const float3 v0_i = d_velocity[i];
    float new_x = (d0_i.z-b)*d0_i.x-d*d0_i.y;
    float new_y = d*d0_i.x + (d0_i.z-b)*d0_i.y;
    float new_z = c+a*d0_i.z-d0_i.z*d0_i.z*d0_i.z/3.-(d0_i.x*d0_i.x+d0_i.y*d0_i.y)*(1+e*d0_i.z)+f*d0_i.z*d0_i.x*d0_i.x*d0_i.x;
    
    d_velocity[i].x = new_x;
    d_velocity[i].y = new_y;
    d_velocity[i].z = new_z;

    positions[i].x = (d0_i.x + d_velocity[i].x * dt);
    positions[i].y = (d0_i.y + d_velocity[i].y * dt);
    positions[i].z = (d0_i.z + d_velocity[i].z * dt);

    d_velocity[i].x = new_x*100;
    d_velocity[i].y = new_y*100;
    d_velocity[i].z = new_z*100;
}
