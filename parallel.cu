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

#define N 16836  // number of particles 
#define BLOCK_SIZE 512 // p <= N/40 
#define G 1.0 // gravitational constant
// #define DT 0.0000001 // time step 
// #define EPSILON 0.00001 // softening parameter
// #define STEPS 1000000 // simulation steps

#define evolt 0.315f

#define DT 0.000001f // time step 
#define EPS2 1e-9f // softening parameter
#define STEPS 1000 // simulation steps
#define L 100.0 // box size

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
 * Generate random initial conditions for N particles using uniform distributions
 */
__host__ void ic_random_uniform(
    int n_particles, 
    double mass_range[2],
    double pos_range[2],
    double vel_range[2],
    float4 *pos, float4 *vel,
    int center_of_mass) {
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

__device__ float3 tileCalculation(float4 myPos, float3 accel) {

    // Shared memory for positions of particles in the tile
    extern __shared__ float4 shPosition[];
    #pragma unroll 128
    for (int i = 0; i < blockDim.x; i++) {
        accel = computeAcceleration(myPos, shPosition[i], accel);
    }
    return accel;
}

__global__ void forceCalculation(void *d_pos, void *d_acc) { 
    
    extern __shared__ float4 shPosition[];

    // assign them to local pointers with type conversion 
    // so they can be indexed as arrays
    float4 *globPos = (float4*)d_pos;
    float4 *globAcc = (float4*)d_acc;
    float4 myPos;
    int i, tile;

    float3 acc = {0.0f, 0.0f, 0.0f};
    int p = blockDim.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    myPos = globPos[gtid];

    for (i = 0, tile = 0; i < N; i += p, tile++) {

        int idx = tile * blockDim.x + threadIdx.x;

        shPosition[threadIdx.x] = globPos[idx];
        
        __syncthreads();
        acc = tileCalculation(myPos, acc);
        __syncthreads();
    }

    // Save the result in global memory for the integration step.
   float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
   globAcc[gtid] = acc4;
}

__global__ void updatePositions(int n, float4 *pos, float4 *vel, float4 *acc, float dt) { 
    
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < n) { 
        pos[i].x += vel[i].x*dt + 0.5 * acc[i].x * dt * dt; 
        pos[i].y += vel[i].y*dt + 0.5 * acc[i].y * dt * dt; 
        pos[i].z += vel[i].z*dt + 0.5 * acc[i].z * dt * dt;  
    } 
}

__global__ void updateVelocities(int n, float4 *vel, float4 *oldAcc, float4 *newAcc, 
                               double dt) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < n) { 
        vel[i].x += 0.5 * (oldAcc[i].x + newAcc[i].x) * dt; 
        vel[i].y += 0.5 * (oldAcc[i].y + newAcc[i].y) * dt; 
        vel[i].z += 0.5 * (oldAcc[i].z + newAcc[i].z) * dt; 
    } 
}

__global__ void computeEnergy(int n, float4 *pos, float4 *vel, float1 *totEn) {
    float1 kinetic = {0.0f};
    float1 potential = {0.0f};
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        #pragma unroll
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

void computePerfStats(double &interactionsPerSecond, double &gflops, float milliseconds, int iterations)
{
    const int flopsPerInteraction = 20;
    interactionsPerSecond = (float)N * (float)N;
    interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
    gflops = interactionsPerSecond * (float)flopsPerInteraction;

}

int returnPosition(int sims) { 
    // Host memory
    clock_t memGen_in = clock();
    float4 *h_pos = (float4*)malloc(N * sizeof(float4));
    float4 *h_vel = (float4*)malloc(N * sizeof(float4));
    float3 *com = (float3*)malloc(sizeof(float3));
    float1 *h_totEn = (float1*)malloc(sizeof(float1));

    // Initialize data
    if (sims > 3) {

        double mass_range[2] = {1.0, 10.0};      // Masses between 1.0 and 10.0
        double pos_range[2] = {-50.0, 50.0};     // Positions between -50.0 and 50.0 --> L = 100.0
        double vel_range[2] = {-1.0, 1.0};       // Velocities between -1.0 and 1.0

        ic_random_uniform(N, mass_range, pos_range, vel_range, h_pos, h_vel, 1);
        clock_t memGen_out = clock();
        double memGen_time = (double)(memGen_out - memGen_in) / CLOCKS_PER_SEC;
        std::cout << "Host mem alloc + body gen: " << memGen_time << " s\n";
    
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

    clock_t memAlloc_in = clock();
    // Device memory - using float4 for positions, velocities and accelerations
    float4 *d_pos, *d_vel, *d_acc_old, *d_acc_new;
    float1 *d_totEn;
    
    // Allocate memory for float4 arrays
    cudaMalloc(&d_pos, N * sizeof(float4));
    cudaMalloc(&d_vel, N * sizeof(float4));
    cudaMalloc(&d_acc_old, N * sizeof(float4));
    cudaMalloc(&d_acc_new, N * sizeof(float4));
    cudaMalloc(&d_totEn, sizeof(float1));
    
    
    // Copy initial data to device
    cudaMemcpy(d_pos, h_pos, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_totEn, h_totEn, sizeof(float1), cudaMemcpyHostToDevice);
    
    // Initialize old acceleration to zero
    cudaMemset(d_acc_old, 0, N * sizeof(float4));
    clock_t memAlloc_out = clock();
    double memAlloc_time = (double)(memAlloc_out - memAlloc_in) / CLOCKS_PER_SEC;
    std::cout << "Device mem alloc: " << memAlloc_time << " s\n";

    /* FILE *fp = fopen("parallel_output.csv", "w");
    if (!fp) {
        printf("Error opening file\n");
        return 1;
    }
    // If the file is empty, write the header.
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "ID,t_step,pos_x,pos_y,pos_z\n");
    }

    FILE *fpe = fopen("parallel_energy.csv", "w");
    if (!fpe) {
        printf("Error opening file\n");
        return 1;
    }
    // If the file is empty, write the header.
    fseek(fpe, 0, SEEK_END);
    if (ftell(fpe) == 0) {
        fprintf(fpe, "t_step,e_tot\n");
    } */

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sharedMemSize = BLOCK_SIZE * sizeof(float4);

    for (int step = 0; step < STEPS; step++) {
        // Calculate forces and update accelerations
        forceCalculation<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_pos, d_acc_new);
        
        // Update positions
        updatePositions<<<numBlocks, BLOCK_SIZE>>>(N, d_pos, d_vel, d_acc_old, DT);
        
        // Update velocities
        updateVelocities<<<numBlocks, BLOCK_SIZE>>>(N, d_vel, d_acc_old, d_acc_new, DT);

        /* computeEnergy<<<numBlocks, BLOCK_SIZE>>>(N, d_pos, d_vel, d_totEn);
        cudaMemcpy(h_totEn, d_totEn, sizeof(float1), cudaMemcpyDeviceToHost); */
        
        // Copy new acceleration to old acceleration
        cudaMemcpy(d_acc_old, d_acc_new, N * sizeof(float4), cudaMemcpyDeviceToDevice);

        cudaMemcpy(h_pos, d_pos, N * sizeof(float4), cudaMemcpyDeviceToHost);
        /* if (step % 100 == 0) {
            fprintf(fpe,"%d,%.6f\n", step, h_totEn->x);
            for (int i = 0; i < N; i++) {
                fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n",
                    i, step, h_pos[i].x, h_pos[i].y, h_pos[i].z);
            }
        } */
    }

    /* fclose(fp);
    fclose(fpe); */

    // Cleanup
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc_old);
    cudaFree(d_acc_new);
    cudaFree(d_totEn);
    
    free(h_pos);
    free(h_vel);
    free(com);
    free(h_totEn);

    return 0;
}

int returnEnergy() {
    // Host memory
    clock_t memGen_in = clock();
    float4 *h_pos = (float4*)malloc(N * sizeof(float4));
    float4 *h_vel = (float4*)malloc(N * sizeof(float4));
    float3 *com = (float3*)malloc(sizeof(float3));
    float1 *h_totEn = (float1*)malloc(sizeof(float1));

    /* // Initialize data
    double mass_range[2] = {1.0, 10.0};      // Masses between 1.0 and 10.0
    double pos_range[2] = {-50.0, 50.0};     // Positions between -50.0 and 50.0 --> L = 100.0
    double vel_range[2] = {-1.0, 1.0};       // Velocities between -1.0 and 1.0

    ic_random_uniform(N, mass_range, pos_range, vel_range, h_pos, h_vel, 1); */

    // two-body sys
    h_pos[0].x = 0.0; h_pos[0].y = 0.0; h_pos[0].z = 0.0; h_pos[0].w = 8.0;
    h_pos[1].x = 0.1; h_pos[1].y = 0.0; h_pos[1].z = 0.0; h_pos[1].w = 2.0;
    h_vel[0].x = 0.0; h_vel[0].y = -2.0; h_vel[0].z = 0.0; h_vel[0].w = 0.0;
    h_vel[1].x = 0.0; h_vel[1].y = 8.0; h_vel[1].z = 0.0; h_vel[1].w = 0.0;

    clock_t memGen_out = clock();
    double memGen_time = (double)(memGen_out - memGen_in) / CLOCKS_PER_SEC;
    std::cout << "Host mem alloc + body gen: " << memGen_time << " s\n";

    clock_t memAlloc_in = clock();
    // Device memory - using float4 for positions, velocities and accelerations
    float4 *d_pos, *d_vel, *d_acc_old, *d_acc_new;
    float1 *d_totEn;
    
    // Allocate memory for float4 arrays
    cudaMalloc(&d_pos, N * sizeof(float4));
    cudaMalloc(&d_vel, N * sizeof(float4));
    cudaMalloc(&d_acc_old, N * sizeof(float4));
    cudaMalloc(&d_acc_new, N * sizeof(float4));
    cudaMalloc(&d_totEn, sizeof(float1));
    
    
    // Copy initial data to device
    cudaMemcpy(d_pos, h_pos, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_totEn, h_totEn, sizeof(float1), cudaMemcpyHostToDevice);
    
    // Initialize old acceleration to zero
    cudaMemset(d_acc_old, 0, N * sizeof(float4));
    clock_t memAlloc_out = clock();
    double memAlloc_time = (double)(memAlloc_out - memAlloc_in) / CLOCKS_PER_SEC;
    std::cout << "Device mem alloc: " << memAlloc_time << " s\n";

    FILE *fp = fopen("parallel_energy.csv", "w");
    if (!fp) {
        printf("Error opening file\n");
        return 1;
    }
    // If the file is empty, write the header.
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "t_step,e_tot\n");
    }


    auto beginTime = std::chrono::high_resolution_clock::now();

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sharedMemSize = BLOCK_SIZE * sizeof(float4);

    clock_t kernel_in = clock();
    for (int step = 0; step < STEPS; step++) {
        // Calculate forces and update accelerations
        forceCalculation<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_pos, d_acc_new);
        
        // Update positions
        updatePositions<<<numBlocks, BLOCK_SIZE>>>(N, d_pos, d_vel, d_acc_old, DT);
        
        // Update velocities
        updateVelocities<<<numBlocks, BLOCK_SIZE>>>(N, d_vel, d_acc_old, d_acc_new, DT);

        // Compute total energy
        computeEnergy<<<numBlocks, BLOCK_SIZE>>>(N, d_pos, d_vel, d_totEn);

        cudaMemcpy(h_totEn, d_totEn, sizeof(float1), cudaMemcpyDeviceToHost);
        
        // Copy new acceleration to old acceleration
        cudaMemcpy(d_acc_old, d_acc_new, N * sizeof(float4), cudaMemcpyDeviceToDevice);
        
        if (step % 100 == 0) {
            fprintf(fp,"%d,%.6f\n", step, h_totEn->x);\
        }
        
    }
    clock_t kernel_out = clock();
    double kernel_time = (double)(kernel_out - kernel_in) / CLOCKS_PER_SEC;
    std::cout << "Kernel time: " << kernel_time << " s\n";

    auto endTime = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime).count();
    
    double gflops = (20.0 * N * N * STEPS) / (1e6 * ms);
    
    std::cout << "Time: " << ms << " ms\n";
    std::cout << "GFLOPS: " << gflops << std::endl;

    fclose(fp);

    // Cleanup
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc_old);
    cudaFree(d_acc_new);
    cudaFree(d_totEn);
    
    free(h_pos);
    free(h_vel);
    free(com);
    free(h_totEn);

    return 0;
}

void runBenchmark(int iterations)
{
    // once without timing to prime the GPU
    returnPosition(N);

    clock_t t_start = clock();  
        returnPosition(N);
    clock_t t_end = clock();  

    float milliseconds = ((float)t_end - (float)t_start) / CLOCKS_PER_SEC * 1000;
    double interactionsPerSecond = 0;
    double gflops = 0;
    computePerfStats(interactionsPerSecond, gflops, milliseconds, iterations);
    
    printf("%d bodies, total time for %d iterations: %0.3f ms\n", 
           N, iterations, milliseconds);
    printf("= %0.3f billion interactions per second\n", interactionsPerSecond);
    printf("= %0.3f GFLOP/s at %d flops per interaction\n", gflops, 20);
    
}

int main() {
    // returnPosition();
    // returnEnergy();
    runBenchmark(STEPS);
    return 0;
} 