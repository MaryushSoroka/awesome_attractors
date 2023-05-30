#include <iostream>

#include <GLFW/glfw3.h>

#include <cuda.h>


#include "attractor.hpp"
#include "nbody.cuh"

double Attractor::step(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities)
{
    double start = glfwGetTime();
    launchKernel(numBlocks, threadsPerBlock, positions, velocities);
    double stop = glfwGetTime();

    return stop - start;
};


void Lorenz::init(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions)
{
    std::cout << "Initializing Lorenz Attractor" << std::endl;
    float bounds_h[6] = {-25., 20., 5., 40., 10., 40.};
    launchInitKernel(numBlocks, threadsPerBlock, positions, bounds_h);
};

void Lorenz::launchKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities)
{
    // std::cout << "Launching Lorentz Kernel" << std::endl;
    launchLorentzKernel(numBlocks, threadsPerBlock, positions, velocities, simulation_speed);
};


void Chen::init(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions)
{
    std::cout << "Initializing Chen Attractor" << std::endl;
    float bounds_h[6] = {-15., 15., -15., 15., 10., 40.};
    launchInitKernel(numBlocks, threadsPerBlock, positions, bounds_h);
};

void Chen::launchKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities)
{
    launchChenKernel(numBlocks, threadsPerBlock, positions, velocities, simulation_speed);
};

void Isawa::init(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions)
{
    std::cout << "Initializing Isawa Attractor" << std::endl;
    float bounds_h[6] = {-2., 2., -2., 2., -2., 0.};
    launchInitKernel(numBlocks, threadsPerBlock, positions, bounds_h);
};

void Isawa::launchKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities)
{
    launchIsawaKernel(numBlocks, threadsPerBlock, positions, velocities, simulation_speed);
};


