#pragma once

#include "glm/glm.hpp"

#include <cuda.h>
#include "nbody.cuh"

class Attractor {
    public:
        glm::vec3 cam_position;
        float cam_speed;        
        float simulation_speed;
        
        double step(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities);

        virtual void init(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions) = 0;
        virtual inline void launchKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities) = 0;
};

class Lorenz: public Attractor {
    public:
        Lorenz(): Attractor()
        {
            cam_position = glm::vec3(1,0,0);
            cam_speed = 0.01;
            simulation_speed = 0.001;
        };

        void init(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions);
        void launchKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities);
};

class Chen: public Attractor {
    public:
        Chen(): Attractor()
        {
            cam_position = glm::vec3(1,0,0);
            cam_speed = 0.008;
            simulation_speed = 0.001;
        };

        void init(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions);
        void launchKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities);
};

class Isawa: public Attractor {
    public:
        Isawa(): Attractor()
        {
            cam_position = glm::vec3(1,0,0);
            cam_speed = 0.004;
            simulation_speed = 0.002;
        };

        void init(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions);
        void launchKernel(unsigned int numBlocks, unsigned int threadsPerBlock, float3* positions, float3* velocities);
};

