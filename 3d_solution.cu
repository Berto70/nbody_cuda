#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h>
#include <time.h>

#define N 20 // number of particles 
#define BLOCK_SIZE 256 
#define G 1.0 // gravitational constant
#define DT 0.0001 // time step 
#define EPSILON 0.00001 // softening parameter
#define STEPS 500000 // simulation steps
#define L 100.0 // box size
#define TILE_SIZE 32  // Size of the tile in shared memory

// Kernel: Compute gravitational accelerations for each particle. 
// Each thread computes the acceleration on one particle due to all others. 
__global__ void computeAccelerationTiled(int n, double *mass, 
                                       double *posX, double *posY, double *posZ, 
                                       double *accX, double *accY, double *accZ) {
    __shared__ double s_mass[TILE_SIZE];
    __shared__ double s_posX[TILE_SIZE];
    __shared__ double s_posY[TILE_SIZE];
    __shared__ double s_posZ[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double ax = 0.0, ay = 0.0, az = 0.0;
    
    // Cache the position of particle i
    double myPosX = (i < n) ? posX[i] : 0.0;
    double myPosY = (i < n) ? posY[i] : 0.0;
    double myPosZ = (i < n) ? posZ[i] : 0.0;
    
    // Loop over tiles
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Collaboratively load tile into shared memory
        int idx = tile * TILE_SIZE + threadIdx.x;
        s_mass[threadIdx.x] = (idx < n) ? mass[idx] : 0.0;
        s_posX[threadIdx.x] = (idx < n) ? posX[idx] : 0.0;
        s_posY[threadIdx.x] = (idx < n) ? posY[idx] : 0.0;
        s_posZ[threadIdx.x] = (idx < n) ? posZ[idx] : 0.0;
        
        __syncthreads();
        
        // Compute partial forces from particles in this tile
        if (i < n) {
            #pragma unroll 32
            for (int j = 0; j < TILE_SIZE && (tile * TILE_SIZE + j) < n; j++) {
                if ((tile * TILE_SIZE + j) != i) {
                    double dx = s_posX[j] - myPosX;
                    double dy = s_posY[j] - myPosY;
                    double dz = s_posZ[j] - myPosZ;
                    
                    double distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
                    double invDistCube = 1.0 / (sqrt(distSqr) * distSqr);
                    
                    double factor = G * s_mass[j] * invDistCube;
                    ax += factor * dx;
                    ay += factor * dy;
                    az += factor * dz;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write final accelerations to global memory
    if (i < n) {
        accX[i] = ax;
        accY[i] = ay;
        accZ[i] = az;
    }
}

// Kernel: Update positions using current velocities and accelerations. 
// We then apply wrap‐around boundaries. 
__global__ void updatePositions(int n, 
                                double *posX, double *posY, double *posZ, 
                                double *velX, double *velY, double *velZ, 
                                double *accX, double *accY, double *accZ,
                                double dt) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < n) { 
        posX[i] += velX[i]*dt + 0.5 * accX[i] * dt * dt; 
        posY[i] += velY[i]*dt + 0.5 * accY[i] * dt * dt; 
        posZ[i] += velZ[i]*dt + 0.5 * accZ[i] * dt * dt; 

        // Wrap-around: ensure the position stays within [0, L) 
        // posX[i] = fmod(posX[i] + L, L); 
        // posY[i] = fmod(posY[i] + L, L);
        // posZ[i] = fmod(posZ[i] + L, L); 
    } 
    
}

// Kernel: Update velocities using the average of old and new accelerations. 
__global__ void updateVelocities(int n, 
                                double *velX, double *velY, double *velZ, 
                                double *oldAccX, double *oldAccY, double *oldAccZ, 
                                double *newAccX, double *newAccY, double *newAccZ, 
                                double dt) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if(i < n) { 
        velX[i] += 0.5 * (oldAccX[i] + newAccX[i]) * dt; 
        velY[i] += 0.5 * (oldAccY[i] + newAccY[i]) * dt; 
        velZ[i] += 0.5 * (oldAccZ[i] + newAccZ[i]) * dt; 
    } 
    
}

int main() { 
    // Allocate host memory for masses, positions, and velocities. 
    double *h_mass = (double*)malloc(N * sizeof(double)); 
    double *h_posX = (double*)malloc(N * sizeof(double)); 
    double *h_posY = (double*)malloc(N * sizeof(double)); 
    double *h_posZ = (double*)malloc(N * sizeof(double)); 
    double *h_velX = (double*)malloc(N * sizeof(double)); 
    double *h_velY = (double*)malloc(N * sizeof(double)); 
    double *h_velZ = (double*)malloc(N * sizeof(double));

    // Initialize synthetic data:
    // masses in [1,10], positions in [0,L] and velocities in [-1,1].
    srand(42);
    for (int i = 0; i < N; i++) {
        h_mass[i] = 1.0; //+ (rand() / (double)RAND_MAX) * 9.0;
        h_posX[i] = (rand() / (double)RAND_MAX); //* L;
        h_posY[i] = (rand() / (double)RAND_MAX); //* L;
        h_posZ[i] = (rand() / (double)RAND_MAX);// * L;
        h_velX[i] = 0;//-1.0 + (rand() / (double)RAND_MAX) * 2.0;
        h_velY[i] = 0;//-1.0 + (rand() / (double)RAND_MAX) * 2.0;
        h_velZ[i] = 0;//-1.0 + (rand() / (double)RAND_MAX) * 2.0;
    }

    // Allocate device memory.
    double *d_mass, *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    double *d_accX_old, *d_accY_old, *d_accZ_old;
    double *d_accX_new, *d_accY_new, *d_accZ_new;
    cudaMalloc((void**)&d_mass, N*sizeof(double));
    cudaMalloc((void**)&d_posX, N*sizeof(double));
    cudaMalloc((void**)&d_posY, N*sizeof(double));
    cudaMalloc((void**)&d_posZ, N*sizeof(double));
    cudaMalloc((void**)&d_velX, N*sizeof(double));
    cudaMalloc((void**)&d_velY, N*sizeof(double));
    cudaMalloc((void**)&d_velZ, N*sizeof(double));
    cudaMalloc((void**)&d_accX_old, N*sizeof(double));
    cudaMalloc((void**)&d_accY_old, N*sizeof(double));
    cudaMalloc((void**)&d_accZ_old, N*sizeof(double));
    cudaMalloc((void**)&d_accX_new, N*sizeof(double));
    cudaMalloc((void**)&d_accY_new, N*sizeof(double));
    cudaMalloc((void**)&d_accZ_new, N*sizeof(double));

    // Copy initial data from host to device.
    cudaMemcpy(d_mass, h_mass, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posX, h_posX, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, h_posY, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, h_posZ, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX, h_velX, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, h_velY, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ, h_velZ, N*sizeof(double), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Compute initial acceleration.
    computeAccelerationTiled<<<numBlocks, BLOCK_SIZE>>>(N, d_mass, d_posX, d_posY, d_posZ,
                                                    d_accX_old, d_accY_old, d_accZ_old);
    cudaDeviceSynchronize();


    // Open a file to save the simulation history.
    FILE *fp = fopen("position_output.csv", "w");
    if (!fp) {
        printf("Error opening file\n");
        return 1;
    }
    // If the file is empty, write the header.
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "ID,t_step,pos_x,pos_y,pos_z\n");
    }

    clock_t time_in = clock();
    // Main simulation loop with history saving.
    for (int step = 0; step < STEPS; step++) {
        // Update positions based on current velocities and acceleration.
        updatePositions<<<numBlocks, BLOCK_SIZE>>>(N, d_posX, d_posY, d_posZ,
                                                    d_velX, d_velY, d_velZ, d_accX_old, d_accY_old, d_accZ_old, DT);
        cudaDeviceSynchronize();

        // Compute new acceleration based on updated positions.
        computeAccelerationTiled<<<numBlocks, BLOCK_SIZE>>>(N, d_mass, d_posX, d_posY, d_posZ,
                                                        d_accX_new, d_accY_new, d_accZ_new);
        cudaDeviceSynchronize();

        // Update velocities using the average of old and new accelerations.
        updateVelocities<<<numBlocks, BLOCK_SIZE>>>(N, d_velX, d_velY, d_velZ,
                                                    d_accX_old, d_accY_old, d_accZ_old,
                                                    d_accX_new, d_accY_new, d_accZ_new, DT);
        cudaDeviceSynchronize();

        // Prepare for the next iteration.
        cudaMemcpy(d_accX_old, d_accX_new, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_accY_old, d_accY_new, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_accZ_old, d_accZ_new, N * sizeof(double), cudaMemcpyDeviceToDevice);

        // Copy the current positions and velocities to host memory.
        cudaMemcpy(h_posX, d_posX, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_posY, d_posY, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_posZ, d_posZ, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_velX, d_velX, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_velY, d_velY, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_velZ, d_velZ, N * sizeof(double), cudaMemcpyDeviceToHost);

        // Save the state for each particle.
        // The columns are: timestep, particleID, posX, posY, posZ, velX, velY, velZ.
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n",
                   i, step, h_posX[i], h_posY[i], h_posZ[i]);
        }
    }
    clock_t time_out = clock();
    printf("Time: %f\n", (double)(time_out - time_in) / CLOCKS_PER_SEC);

    fclose(fp);

    // Free device memory.
    cudaFree(d_mass);
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);
    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velZ);
    cudaFree(d_accX_old);
    cudaFree(d_accY_old);
    cudaFree(d_accZ_old);
    cudaFree(d_accX_new);
    cudaFree(d_accY_new);
    cudaFree(d_accZ_new);

    // Free host memory.
    free(h_mass);
    free(h_posX);
    free(h_posY);
    free(h_posZ);
    free(h_velX);
    free(h_velY);
    free(h_velZ);

    return 0;
}