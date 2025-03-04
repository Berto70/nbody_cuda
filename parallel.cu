/* 
 * N-body simulator - Parallelized with CUDA
 * Author: Gabriele Bertinelli
 * Modern Computing for Physics course - University of Padova
 * 2024 - 2025
 */

// How to compile the script:
// nvcc nbody_cuda/parallel.cu -o nbody_cuda/bin/parallel -O3 --use_fast_math -arch=sm_75
// -O3: optimization level 3 (max)
// --use_fast_math: enable fast math operations
// -arch=sm_75: specify the GPU architecture

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <memory.h>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <time.h>
#include <chrono>


//##############################################################################
//# CONSTANTS
//##############################################################################
#define N 65536 // number of particles 
#define BLOCK_SIZE 128 // p <= N/40 
#define G 1.0 // gravitational constant

#define evolt 0.315f
#define DT 0.000001f // time step 
#define EPS2 1e-9f // softening parameter
#define STEPS 1000 // simulation steps
#define L 100.0 // box size


//##############################################################################
//# DEVICE FUNCTIONS AND KERNELS 
//##############################################################################

/**
 * @brief Compute gravitational acceleration between two bodies using Newton's law
 * 
 * This device function calculates the gravitational acceleration exerted by body j
 * on body i, and adds it to the current acceleration of body i. It implements
 * the pairwise interaction for an N-body gravitational simulation.
 * 
 * The calculation includes a softening factor (EPS2) to prevent numerical 
 * instabilities when bodies are very close to each other.
 * 
 * FLOP count: 20 floating-point operations per call.
 * We use CUDA's float4 data type for descriptions. Using float4 (instead of float3) 
 * data allows coalesced memory access to the arrays of data in device memory, 
 * resulting in efficient memory requests and transfers. 
 * 
 * @param bi Position and mass of body i (x, y, z, mass) - the current body
 * @param bj Position and mass of body j (x, y, z, mass) - the interacting body
 * @param ai Current acceleration vector of body i
 * @return Updated acceleration vector of body i after adding contribution from body j
 * 
 * @note This function is executed on the device (GPU).
 */
__device__ float3 computeAcceleration(float4 bi, float4 bj, float3 ai) {
    float3 r;

    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    
    // ||r_ij||^2 + eps^2 [6 FLOPS]
    float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + EPS2;

    // 1/distSqr^(3/2) [4 FLOPS]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = rsqrtf(distSixth);
    
    // m_j * 1/distSqr^(3/2) [1 FLOP]
    float s = bj.w * invDistCube;
    
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += s * r.x;
    ai.y += s * r.y;
    ai.z += s * r.z;
    
    return ai; // tot [20 FLOPS]
}

/**
 * @brief Calculates the acceleration contribution from all particles in the current tile
 *
 * This device function computes the gravitational contribution to a particle's acceleration 
 * from all other particles in the shared memory tile.
 * 
 * A tile is evaluated as p threads performing the same sequence of operations 
 * on different data. Each thread updates the acceleration of one body as a result of 
 * its interaction with p other bodies. 
 * The function load p body descriptions from device memory into shared memory 
 * provided to each thread block by CUDA. Each thread block evaluates p successive 
 * interactions. The result of the tile calculation is p updated acceleration.
 * 
 * @param[in] myPos The position of the current particle (x, y, z, mass)
 * @param[out] accel The current accumulated acceleration vector of the particle
 * @return Updated acceleration vector after accounting for all particles in the tile
 *
 * @note Unrolling the loop can reduce the number of overhead instructions.
 * 
 * @note This function requires shared memory to be allocated and populated with 
 *       particle positions before being called. The shared memory size must be at 
 *       least sizeof(float4) * blockDim.x.
 */
template<bool unrollLoop>
__device__ float3 tileCalculation(float4 myPos, float3 accel) {

    // Shared memory for positions of particles in the tile
    // Memory shared across all threads in a block
    extern __shared__ float4 shPosition[];
    
    if (unrollLoop) {
        #pragma unroll 16 // Unroll the loop to reduce overhead
        // Iterate through all p in this tile (tile size = blockDim.x)
        // Each thread processes interactions with all p in shared memory
        for (int i = 0; i < blockDim.x; i++) {
            accel = computeAcceleration(myPos, shPosition[i], accel);
        }
    }
    else {
        for (int i = 0; i < blockDim.x; i++) {
            accel = computeAcceleration(myPos, shPosition[i], accel);
        }
    }
    
    // Return the accumulated acceleration in the tile
    return accel;
}

/**
 * @brief CUDA kernel that calculates the gravitational forces between particles in an N-body simulation.
 *
 * This kernel computes the acceleration for each particle due to gravitational interactions
 * with all other particles in the simulation. It uses a tiled approach with shared memory
 * to improve performance by reducing global memory accesses.
 * 
 * In a thread block there are N/p tiles, with p threads computing the forces 
 * on p bodies. A thread block reloads its shared memory every p steps to share 
 * p positions of data. Each thread in a block computes all N interactions for one body.
 * The code calculates N-body forces for a thread block. 
 * 
 * The loop over the tiles requires two synchronization points:  
 * 
 * 1. The first synchronization ensures that all shared memory locations 
 * are populated before the computation proceeds;  
 * 
 * 2. the second ensures that all threads finish their computation before 
 * advancing to the next tile. 
 *
 * Each thread computes the net acceleration for one particle by:
 * 
 * 1. Loading its assigned particle position
 * 
 * 2. Processing all other particles in tiles using shared memory
 * 
 * 3. Computing partial accelerations for each tile
 * 
 * 4. Accumulating the final acceleration vector
 *
 * @param[in] d_pos Pointer to particle positions in global memory (as float4 array where x,y,z = position, w = mass)
 * @param[out] d_acc Pointer to acceleration output array in global memory (as float4 array)
 *
 * @note Requires shared memory allocation of blockDim.x * sizeof(float4) bytes at kernel launch
 * @note N must be defined as a global constant representing the total number of bodies
 * @note This function is executed on the device (GPU).
 */
template<bool unrollLoop>
__global__ void forceCalculation(void *d_pos, void *d_acc) { 
    
    // Shared memory for storing particle data for the current tile
    extern __shared__ float4 shPosition[];

    // Cast void pointers to typed pointers for array indexing
    // so they can be indexed as arrays
    float4 *globPos = (float4*)d_pos;
    float4 *globAcc = (float4*)d_acc;
    float4 myPos;
    int i, tile;

    // Initialize acc accumulator for this thread's particle
    float3 acc = {0.0f, 0.0f, 0.0f};
    // Number of particles in a tile
    int p = blockDim.x;
    // Global index - unique identifier for each particle across all blocks
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // Load thread's assigned particle position
    myPos = globPos[gtid];

    // Process all particles in tiles
    for (i = 0, tile = 0; i < N; i += p, tile++) {

        // Calculate the global index for this thread's 
        // assigned particle in the current tile
        int idx = tile * blockDim.x + threadIdx.x;

        // Each thread loads one particle from the current tile into shared memory
        shPosition[threadIdx.x] = globPos[idx];
        
        // Ensure all threads have loaded their data before computation
        __syncthreads();
        acc = tileCalculation<unrollLoop>(myPos, acc);
        __syncthreads();
    }

    // Save the result in global memory for the integration step
    // Convert float3 acceleration to float4
   float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
   globAcc[gtid] = acc4;
}

/**
 * @brief CUDA kernel that updates positions of particles in an N-body simulation
 *
 * This kernel updates the positions of n particles based on their velocities and
 * accelerations using the Verlet integration method.
 * 
 * @param[in] n Number of bodies
 * @param pos Array of particle positions (and mass) as float4 where x,y,z are positions
 * @param[in] vel Array of particle velocities as float4 where x,y,z are velocity components
 * @param[in] acc Array of particle accelerations as float4 where x,y,z are acceleration components
 * @param[in] dt Time step for the simulation
 *
 * @note Each thread processes one particle, because the integration scales as O(N), 
 * thus this impacts less on the overall performance.
 * @note The code uses the control construct if (i < n) to handle boundary 
 * conditions when mapping threads to data. This is usually because the total 
 * number of threads needs to be a multiple of the block size whereas the size 
 * of the data can be an arbitrary number.
 * @note This function is executed on the device (GPU).
 */
__global__ void updatePositions(int n, float4 *pos, float4 *vel, float4 *acc, float dt) { 
    
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    // Handles the case where the number of particles is not a multiple of the block size
    if (i < n) { 
        pos[i].x += vel[i].x*dt + 0.5 * acc[i].x * dt * dt; 
        pos[i].y += vel[i].y*dt + 0.5 * acc[i].y * dt * dt; 
        pos[i].z += vel[i].z*dt + 0.5 * acc[i].z * dt * dt;  
    } 
}

/**
 * @brief CUDA kernel that updates velocities of particles in an N-body simulation
 *
 * This kernel updates the velocity of each body based on old and new accelerations
 * using the velocity Verlet integration method.
 *
 * @param[in] n Number of bodies
 * @param vel Array of float4 values representing velocities (x,y,z components)
 * @param[in] oldAcc Array of float4 values representing accelerations at time t
 * @param[in] newAcc Array of float4 values representing accelerations at time t+dt
 * @param[in] dt Time step size
 * 
 * @note Each thread processes one particle, because the integration scales as O(N), 
 * thus this impacts less on the overall performance.
 * @note The code uses the control construct if (i < n) to handle boundary 
 * conditions when mapping threads to data. This is usually because the total 
 * number of threads needs to be a multiple of the block size whereas the size 
 * of the data can be an arbitrary number.
 * @note This function is executed on the device (GPU).
 */
__global__ void updateVelocities(int n, float4 *vel, float4 *oldAcc, float4 *newAcc, 
                               double dt) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    // Handles the case where the number of particles is not a multiple of the block size
    if (i < n) { 
        vel[i].x += 0.5 * (oldAcc[i].x + newAcc[i].x) * dt; 
        vel[i].y += 0.5 * (oldAcc[i].y + newAcc[i].y) * dt; 
        vel[i].z += 0.5 * (oldAcc[i].z + newAcc[i].z) * dt; 
    } 
}

/**
 * @brief CUDA kernel to compute the total energy of an N-body system
 * 
 * This kernel calculates both kinetic and potential energy for each particle
 * and accumulates them to get the total energy of the system. Each thread
 * processes one particle's energy contribution.
 * 
 * @param[in] n Number of bodies
 * @param[in] pos Array of float4 values where (x,y,z) is the position and w is the mass of each particle
 * @param[in] vel Array of float4 values where (x,y,z) is the velocity of each particle
 * @param[out] totEn Pointer to store the resulting total energy
 * 
 * @note This function is executed on the device (GPU).
 */
__global__ void computeEnergy(int n, float4 *pos, float4 *vel, float1 *totEn) {
    float1 kinetic = {0.0f};
    float1 potential = {0.0f};
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        #pragma unroll 16
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float3 r;
                r.x = pos[j].x - pos[i].x;
                r.y = pos[j].y - pos[i].y;
                r.z = pos[j].z - pos[i].z;
                
                float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
                float distSixth = distSqr * distSqr * distSqr;
                potential.x -= (pos[i].w * pos[j].w) / sqrt(distSixth);
            }
        }
    }

    kinetic.x += 0.5 * pos[i].w * (vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z);
    totEn->x = kinetic.x + potential.x;
}


//##############################################################################
//# HOST FUNCTIONS
//##############################################################################

/**
 * @brief Computes the center of mass for a system of bodies
 *
 * This function calculates the center of mass for a system of N bodies by computing
 * the weighted average of positions based on the mass of each body. 
 *
 * @param[in] pos Array of float4 values where x,y,z represent positions and w represents mass
 * @param[out] com Pointer to float3 where the calculated center of mass will be stored
 *
 * @note This function is executed on the host (CPU).
 */
__host__ void compute_com(float4 *pos, float3 *com) {
    
    double total_mass = 0.0;
    
    for (int i = 0; i < N; i++) {
        total_mass += pos[i].w;
        com->x += pos[i].w * pos[i].x;
        com->y += pos[i].w * pos[i].y;
        com->z += pos[i].w * pos[i].z;
    }
    
    com->x /= total_mass;
    com->y /= total_mass;
    com->z /= total_mass;
}

/**
 * @brief Initializes particle positions and velocities with uniform random values
 *
 * This function generates a random N-body system with particles distributed 
 * uniformly within the specified ranges. Each particle is assigned a random mass,
 * position, and velocity within the provided bounds. If requested, the system 
 * can be centered at its center of mass.
 *
 * @param[in] n_particles Number of particles to initialize
 * @param[in] mass_range Array of two doubles specifying [min, max] mass range
 * @param[in] pos_range Array of two doubles specifying [min, max] position range for all dimensions
 * @param[in] vel_range Array of two doubles specifying [min, max] velocity range for all dimensions
 * @param[out] pos Pointer to array of float4 values where positions will be stored (x,y,z are position, w is mass)
 * @param[out] vel Pointer to array of float4 values where velocities will be stored (x,y,z are velocity, w unused)
 * @param[in] center_of_mass If 1, shifts all positions to center the system at the center of mass
 *
 * @note Uses a fixed random seed (42) for reproducible results.
 * @note This function is executed on the host (CPU).
 */
__host__ void ic_random_uniform(int n_particles, 
                                double mass_range[2],
                                double pos_range[2],
                                double vel_range[2],
                                float4 *pos, float4 *vel, int center_of_mass) {
    // Initialize random number generator
    srand(42);
    
    // Generate uniform random values within ranges
    for (int i = 0; i < n_particles; i++) {
        // Mass
        pos[i].w = mass_range[0] + (rand() / (double)RAND_MAX) * (mass_range[1] - mass_range[0]);
        
        // Position
        pos[i].x = pos_range[0] + (rand() / (double)RAND_MAX) * (pos_range[1] - pos_range[0]);
        pos[i].y = pos_range[0] + (rand() / (double)RAND_MAX) * (pos_range[1] - pos_range[0]);
        pos[i].z = pos_range[0] + (rand() / (double)RAND_MAX) * (pos_range[1] - pos_range[0]);
        
        // Velocity
        vel[i].x = vel_range[0] + (rand() / (double)RAND_MAX) * (vel_range[1] - vel_range[0]);
        vel[i].y = vel_range[0] + (rand() / (double)RAND_MAX) * (vel_range[1] - vel_range[0]);
        vel[i].z = vel_range[0] + (rand() / (double)RAND_MAX) * (vel_range[1] - vel_range[0]);
        vel[i].w = 0.0f;  // Not used for velocity
    }

    // If 1, center the system by subtracting the CoM
    if (center_of_mass) {
        float3 com = {0.0f, 0.0f, 0.0f};
        compute_com(pos, &com);
        
        // Shift positions to center of mass frame
        for (int i = 0; i < n_particles; i++) {
            pos[i].x -= com.x;
            pos[i].y -= com.y;
            pos[i].z -= com.z;
        }
    }
}

/**
 * @brief Calculates performance statistics for the N-body simulation.
 *
 * This function computes two performance metrics:
 * 1. Interactions per second (in billions) - how many body-body interactions are computed per second
 * 2. GFLOPS - floating point operations per second (in billions)
 *
 * The calculation assumes that each body-body interaction requires 20 floating-point operations.
 * Performance is based on the total number of bodies (N), iteration count, and execution time.
 *
 * @param[out] interactionsPerSecond Computed interactions per second (in billions)
 * @param[out] gflops Computed GFLOPS (floating point operations per second in billions)
 * @param[in] milliseconds Total execution time in milliseconds
 * @param[in] iterations Number of simulation iterations performed
 * 
 * @note This function is executed on the host (CPU).
 */
__host__ void computePerfStats(double &interactionsPerSecond, double &gflops, 
                                float milliseconds) {
    const int flopsPerInteraction = 20;
    interactionsPerSecond = (float)N * (float)N;
    interactionsPerSecond *= 1e-9 * STEPS * 1000 / milliseconds;
    gflops = interactionsPerSecond * (float)flopsPerInteraction;

}

//##############################################################################
//# MAIN FUNCTIONS
//##############################################################################

/**
 * @brief Simulates the time evolution of an N-body system using CUDA.
 *
 * This function handles the complete N-body simulation workflow:
 * 1. Allocates host and device memory for particle data
 * 2. Initializes particle positions and velocities based on the simulation type
 * 3. Executes the time integration using velocity Verlet algorithm in parallel on GPU
 * 4. Optionally saves simulation data and energy values to output files
 * 5. Cleans up memory before exiting
 *
 * The simulation can be initialized in different ways:
 * - Random uniform distribution (sims > 3)
 * - Two-body system (sims = 2), set to have e=0.0 and rp=0.1
 * - Three-body system (sims = 3). It's the problem set by Lagrange: 
 * three masses in a equatorial triangle
 *
 * @param[in] sims Simulation type:
 *             - sims > 3: Random uniform distribution
 *             - sims = 2: Two-body system
 *             - sims = 3: Three-body system
 * @param[in] save_data Whether to save particle positions to output CSV file (1=yes, 0=no)
 * @param[in] energy Whether to compute and save total energy (1=yes, 0=no)
 * @param[in] save_steps Frequency of saving data (save every save_steps iterations)
 *
 * @note Writes output to "parallel_output.csv" when save_data=1
 * @note Writes energy values to "parallel_energy.csv" when both save_data=1 and energy=1
 */
template<bool unrollLoop>
int evolveSystem(int sims, int save_data, int energy, int save_steps) { 
    // Host memory
    // clock_t memGen_in = clock();
    float4 *h_pos = (float4*)malloc(N * sizeof(float4));
    float4 *h_vel = (float4*)malloc(N * sizeof(float4));
    float3 *com = (float3*)malloc(sizeof(float3));

    FILE *fp = NULL;
    FILE *fpe = NULL;
    float1 *h_totEn = NULL;
    float1 *d_totEn = NULL;

    // Initialize data
    if (sims > 3) {

        double mass_range[2] = {1.0, 10.0};      // Masses between 1.0 and 10.0
        double pos_range[2] = {-50.0, 50.0};     // Positions between -50.0 and 50.0 --> L = 100.0
        double vel_range[2] = {-1.0, 1.0};       // Velocities between -1.0 and 1.0

        ic_random_uniform(N, mass_range, pos_range, vel_range, h_pos, h_vel, 1);
        // clock_t memGen_out = clock();
        // double memGen_time = (double)(memGen_out - memGen_in) / CLOCKS_PER_SEC;
        // std::cout << "Host mem alloc + body gen: " << memGen_time << " s\n";
    
    } else if (sims == 2) {

        // two-body sys
        h_pos[0].x = 0.0; h_pos[0].y = 0.0; h_pos[0].z = 0.0; h_pos[0].w = 8.0;
        h_pos[1].x = 0.1; h_pos[1].y = 0.0; h_pos[1].z = 0.0; h_pos[1].w = 2.0;
        h_vel[0].x = 0.0; h_vel[0].y = -2.0; h_vel[0].z = 0.0; h_vel[0].w = 0.0;
        h_vel[1].x = 0.0; h_vel[1].y = 8.0; h_vel[1].z = 0.0; h_vel[1].w = 0.0;

    } else if (sims == 3) {

        // three-body sys
        h_pos[0].x = 1.0; h_pos[0].y = 3.0; h_pos[0].z = 0.0; h_pos[0].w = 3.0;
        h_pos[1].x = -2.0; h_pos[1].y = -1.0; h_pos[1].z = 0.0; h_pos[1].w = 4.0;
        h_pos[2].x = 1.0; h_pos[2].y = -1.0; h_pos[2].z = 0.0; h_pos[2].w = 5.0;
        h_vel[0].x = 0.0; h_vel[0].y = 0.0; h_vel[0].z = 0.0; h_vel[0].w = 0.0;
        h_vel[1].x = 0.0; h_vel[1].y = 0.0; h_vel[1].z = 0.0; h_vel[1].w = 0.0;
        h_vel[2].x = 0.0; h_vel[2].y = 0.0; h_vel[2].z = 0.0; h_vel[2].w = 0.0;
    }

    // clock_t memAlloc_in = clock();
    // Device memory - using float4 for positions, velocities and accelerations
    float4 *d_pos, *d_vel, *d_acc_old, *d_acc_new;
    
    // Allocate memory for float4 arrays
    cudaMalloc(&d_pos, N * sizeof(float4));
    cudaMalloc(&d_vel, N * sizeof(float4));
    cudaMalloc(&d_acc_old, N * sizeof(float4));
    cudaMalloc(&d_acc_new, N * sizeof(float4));     
    
    // Copy initial data to device
    cudaMemcpy(d_pos, h_pos, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, N * sizeof(float4), cudaMemcpyHostToDevice);
    
    if (energy) {
        float1 *h_totEn = (float1*)malloc(sizeof(float1));
        float1 *d_totEn;
        cudaMalloc(&d_totEn, sizeof(float1));
        cudaMemcpy(d_totEn, h_totEn, sizeof(float1), cudaMemcpyHostToDevice);
    }
    
    // Initialize old acceleration to zero
    cudaMemset(d_acc_old, 0, N * sizeof(float4));
    // clock_t memAlloc_out = clock();
    // double memAlloc_time = (double)(memAlloc_out - memAlloc_in) / CLOCKS_PER_SEC;
    // std::cout << "Device mem alloc: " << memAlloc_time << " s\n";

    if (save_data) {
        FILE *fp = fopen("parallel_output.csv", "w");
        if (!fp) {
            printf("Error opening file\n");
            return 1;
        }
        // If the file is empty, write the header.
        fseek(fp, 0, SEEK_END);
        if (ftell(fp) == 0) {
            fprintf(fp, "ID,t_step,pos_x,pos_y,pos_z\n");
        }
    }       

    if (save_data && energy) {
        FILE *fpe = fopen("parallel_energy.csv", "w");
        if (!fpe) {
            printf("Error opening file\n");
            return 1;
        }
        // If the file is empty, write the header.
        fseek(fpe, 0, SEEK_END);
        if (ftell(fpe) == 0) {
            fprintf(fpe, "t_step,e_tot\n");
        }
    }

    // KERNELS

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sharedMemSize = BLOCK_SIZE * sizeof(float4);

    for (int step = 0; step < STEPS; step++) {
        // Calculate forces and update accelerations
        forceCalculation<unrollLoop><<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_pos, d_acc_new);
        
        // Update positions
        updatePositions<<<numBlocks, BLOCK_SIZE>>>(N, d_pos, d_vel, d_acc_old, DT);
        
        // Update velocities
        updateVelocities<<<numBlocks, BLOCK_SIZE>>>(N, d_vel, d_acc_old, d_acc_new, DT);

        // Compute total energy
        if (energy) {
            computeEnergy<<<numBlocks, BLOCK_SIZE>>>(N, d_pos, d_vel, d_totEn);
        }
        
        // Copy new acceleration to old acceleration
        cudaMemcpy(d_acc_old, d_acc_new, N * sizeof(float4), cudaMemcpyDeviceToDevice);
        
        if (save_data) {
            if (step % save_steps == 0) {
                if (energy && save_data) {
                    // Copy total energy to host and write to file
                    cudaMemcpy(h_totEn, d_totEn, sizeof(float1), cudaMemcpyDeviceToHost);
                    fprintf(fpe,"%d,%.6f\n", step, h_totEn->x);
                }   
                for (int i = 0; i < N; i++) {
                    // Copy position to host and write to file
                    cudaMemcpy(h_pos, d_pos, N * sizeof(float4), cudaMemcpyDeviceToHost);
                    fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n",
                        i, step, h_pos[i].x, h_pos[i].y, h_pos[i].z);
                }
            }
        }
    }

    if (save_data) {
        fclose(fp);
    }
    if (save_data && energy) {
        fclose(fpe);
    }

    // Cleanup
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc_old);
    cudaFree(d_acc_new);
    
    free(h_pos);
    free(h_vel);
    free(com);
    
    if (energy) {
        cudaFree(d_totEn);
        free(h_totEn);
    }

    return 0;
}

/**
 * @brief Executes the N-body simulation benchmark
 *
 * This function executes the N-body simulation to measure performance.
 * It first runs one iteration without timing to prime the GPU (warm-up),
 * then executes the actual timed run.
 * After the timed run, it calculates and reports performance statistics 
 * including execution time, interactions per second, and GFLOPS.
 *
 * Priming the GPU helps remove any one-time initialization overhead such as \
 * context setup, lazy memory allocation, and caching. This ensures that the timed 
 * benchmark run reflects steady-state performance rather than being skewed 
 * by startup delays.
 *
 */
/* template<bool unrollLoop>
 void runBenchmark() {
        
    // once without timing to prime the GPU
    evolveSystem<unrollLoop>(N,0,0,100);

    clock_t t_start = clock();  
    evolveSystem<unrollLoop>(N,0,0,100);
    clock_t t_end = clock();  

    float milliseconds = ((float)t_end - (float)t_start) / CLOCKS_PER_SEC * 1000;
    double interactionsPerSecond = 0;
    double gflops = 0;
    computePerfStats(interactionsPerSecond, gflops, milliseconds);
    
    //printf("%d bodies, total time for %d iterations: %0.3f ms\n", 
           //N, STEPS, milliseconds);
    printf("= %0.3f billion interactions per second\n", interactionsPerSecond);
    printf("= %0.3f GFLOP/s at %d flops per interaction\n\n", gflops, 20);
    
}
 */
template<bool unrollLoop>
void runBenchmark() {
        
    // once without timing to prime the GPU
    evolveSystem<unrollLoop>(N,0,0,100);

    clock_t t_start = clock();  
    evolveSystem<unrollLoop>(N,0,0,100);
    clock_t t_end = clock();  

    float milliseconds = ((float)t_end - (float)t_start) / CLOCKS_PER_SEC * 1000;
    double interactionsPerSecond = 0;
    double gflops = 0;
    computePerfStats(interactionsPerSecond, gflops, milliseconds);

    FILE *fp = fopen("nbody_cuda/data/timer.csv", "a");
    if (!fp) {
        printf("Error opening file\n");
        return;
    }
    // If the file is empty, write the header.
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "n_bodies,time\n");
    }
    fprintf(fp, "%d,%0.3f\n", N, milliseconds);
    fclose(fp);
    
    //printf("%d bodies, total time for %d iterations: %0.3f ms\n", 
           //N, STEPS, milliseconds);
    // printf("= %0.3f billion interactions per second\n", interactionsPerSecond);
    // printf("= %0.3f GFLOP/s at %d flops per interaction\n\n", gflops, 20);
    
}

/**
 * @brief Main funtion for the CUDA N-body simulation script.
 * 
 * This function executies the N-body simulation.
 * The user can switch between evolving the system, through evolveSystem(), and
 * running the benchmark, through runBenchmark(), and choosing the number of
 * iterations to run.
 * 
 */
int main() {

    // evolveSystem<false>();
    
    for (int i = 0; i < 11; i++) {
        runBenchmark<true>();
    }

    return 0;
} 