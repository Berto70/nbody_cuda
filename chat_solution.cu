#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h>
#include <time.h>

#define N 255 // number of particles 
#define BLOCK_SIZE 256 
#define G 1.0 // gravitational constant
#define DT 0.0001 // time step 
// #define EPSILON 0.00001 // softening parameter
#define EPSILON 0.0032 // softening parameter
#define STEPS 500000 // simulation steps
#define L 100.0 // box size



// dim3 dimGrid(ceil(N/BLOCK_SIZE), 1, 1);
// dim3 dimBlock(BLOCK_SIZE, 1, 1);

void compute_com(double *mass, double *posX, double *posY, double *posZ, double *com) {
    double total_mass = 0.0;
    com[0] = com[1] = com[2] = 0.0;
    
    for (int i = 0; i < N; i++) {
        total_mass += mass[i];
        com[0] += mass[i] * posX[i];
        com[1] += mass[i] * posY[i];
        com[2] += mass[i] * posZ[i];
    }
    
    com[0] /= total_mass;
    com[1] /= total_mass;
    com[2] /= total_mass;
}

/**
 * Generate random initial conditions for N particles using uniform distributions
 */
void ic_random_uniform(
    int n_particles, 
    double mass_range[2],
    double pos_range[2],
    double vel_range[2],
    double* h_mass,
    double* h_posX, double* h_posY, double* h_posZ,
    double* h_velX, double* h_velY, double* h_velZ,
    int center_of_mass) {
    // Initialize random number generator
    srand(time(NULL));
    
    // Generate uniform random values within ranges
    for (int i = 0; i < n_particles; i++) {
        h_mass[i] = mass_range[0] + (rand() / (double)RAND_MAX) * (mass_range[1] - mass_range[0]);
        
        h_posX[i] = pos_range[0] + (rand() / (double)RAND_MAX) * (pos_range[1] - pos_range[0]);
        h_posY[i] = pos_range[0] + (rand() / (double)RAND_MAX) * (pos_range[1] - pos_range[0]);
        h_posZ[i] = pos_range[0] + (rand() / (double)RAND_MAX) * (pos_range[1] - pos_range[0]);
        
        h_velX[i] = vel_range[0] + (rand() / (double)RAND_MAX) * (vel_range[1] - vel_range[0]);
        h_velY[i] = vel_range[0] + (rand() / (double)RAND_MAX) * (vel_range[1] - vel_range[0]);
        h_velZ[i] = vel_range[0] + (rand() / (double)RAND_MAX) * (vel_range[1] - vel_range[0]);
    }

    // If 1, center the system by subtracting the CoM
    if (center_of_mass) {
        double com[3];
        compute_com(h_mass, h_posX, h_posY, h_posZ, com);
        
        // Shift positions to center of mass frame
        for (int i = 0; i < n_particles; i++) {
            h_posX[i] -= com[0];
            h_posY[i] -= com[1];
            h_posZ[i] -= com[2];
        }
    }
}

__global__ void computeAcceleration(int n, double *mass, 
                                  double *posX, double *posY, double *posZ, 
                                  double *accX, double *accY, double *accZ) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < n) { 
        double ax = 0.0, ay = 0.0, az = 0.0; 
        
        for (int j = 0; j < n; j++) { 
            if (i != j) {
                double dx = posX[j] - posX[i]; 
                double dy = posY[j] - posY[i]; 
                double dz = posZ[j] - posZ[i]; 

                double distSqr = dx*dx + dy*dy + dz*dz + EPSILON*EPSILON; 
                double distCube = distSqr * distSqr * distSqr;
                double invDistCube = 1.0 / sqrtf(distCube); 
                ax += G * mass[j] * dx * invDistCube; 
                ay += G * mass[j] * dy * invDistCube; 
                az += G * mass[j] * dz * invDistCube; 
            } 
        } 
        
        accX[i] = ax; 
        accY[i] = ay; 
        accZ[i] = az; 
    } 
}

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
    } 
}

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
    // Host memory
    double *h_mass = (double*)malloc(N * sizeof(double)); 
    double *h_posX = (double*)malloc(N * sizeof(double)); 
    double *h_posY = (double*)malloc(N * sizeof(double)); 
    double *h_posZ = (double*)malloc(N * sizeof(double)); 
    double *h_velX = (double*)malloc(N * sizeof(double)); 
    double *h_velY = (double*)malloc(N * sizeof(double)); 
    double *h_velZ = (double*)malloc(N * sizeof(double));
    double *com = (double*)malloc(3 * sizeof(double)); // Add com allocation

    // Initialize data
    double mass_range[2] = {1.0, 10.0};      // Masses between 1.0 and 10.0
    double pos_range[2] = {-50.0, 50.0};     // Positions between -50.0 and 50.0 --> L = 100.0
    double vel_range[2] = {-1.0, 1.0};       // Velocities between -1.0 and 1.0

    ic_random_uniform(N, mass_range, pos_range, vel_range, 
                      h_mass, h_posX, h_posY, h_posZ, 
                      h_velX, h_velY, h_velZ, 1);

    // two-body sys

    /* h_mass[0] = 8.0; h_mass[1] = 2.0;
    h_posX[0] = 0.0; h_posY[0] = 0.0;
    h_posX[1] = 0.1; h_posY[1] = 0.0;
    h_velX[0] = 0.0; h_velY[0] = -2.0;
    h_velX[1] = 0.0; h_velY[1] = 8.0; */

    // three-body sys

    /* h_mass[0]=3;h_mass[1]=4;h_mass[2]=5;
    h_posX[0]=1;h_posX[1]=-2;h_posX[2]=1;
    h_posY[0]=3;h_posY[1]=-1;h_posY[2]=-1;
    h_posZ[0]=h_posZ[1]=h_posZ[2]=0;
    h_velX[0]=h_velX[1]=h_velX[2]=0;
    h_velY[0]=h_velY[1]=h_velY[2]=0;
    h_velZ[0]=h_velX[1]=h_velX[2]=0; */

    /* // Calculate centre-of-mass and shift positions accordingly
    double *com = (double*)malloc(N * sizeof(double));
    compute_com(h_mass, h_posX, h_posY, h_posZ, com);
    for (int i = 0; i < N; i++) {
        h_posX[i] -= com[0];
        h_posY[i] -= com[1];
        h_posZ[i] -= com[2];
    } */

    // Device memory
    double *d_mass, *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    double *d_accX_old, *d_accY_old, *d_accZ_old;
    double *d_accX_new, *d_accY_new, *d_accZ_new;

    // Array of pointers to hold all device data
    double **devicePtrs[] = {&d_mass, 
        &d_posX, &d_posY, &d_posZ, 
        &d_velX, &d_velY, &d_velZ,
        &d_accX_old, &d_accY_old, &d_accZ_old,
        &d_accX_new, &d_accY_new, &d_accZ_new};

    // Allocate memory for all arrays in a loop
    for (int i = 0; i < sizeof(devicePtrs)/sizeof(devicePtrs[0]); i++) {
        cudaMalloc(devicePtrs[i], N * sizeof(double));
    }

    // Copy initial data
    cudaMemcpy(d_mass, h_mass, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posX, h_posX, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, h_posY, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, h_posZ, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX, h_velX, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, h_velY, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ, h_velZ, N * sizeof(double), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

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

    // Main simulation loop
    clock_t time_in = clock();
    for (int step = 0; step < STEPS; step++) {
        updatePositions<<<numBlocks, BLOCK_SIZE>>>(N, d_posX, d_posY, d_posZ,
                                                 d_velX, d_velY, d_velZ, 
                                                 d_accX_old, d_accY_old, d_accZ_old, DT);
        
        computeAcceleration<<<numBlocks, BLOCK_SIZE>>>(N, d_mass, d_posX, d_posY, d_posZ,
                                                      d_accX_new, d_accY_new, d_accZ_new);
        
        updateVelocities<<<numBlocks, BLOCK_SIZE>>>(N, d_velX, d_velY, d_velZ,
                                                   d_accX_old, d_accY_old, d_accZ_old,
                                                   d_accX_new, d_accY_new, d_accZ_new, DT);
        
        cudaMemcpy(d_accX_old, d_accX_new, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_accY_old, d_accY_new, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_accZ_old, d_accZ_new, N * sizeof(double), cudaMemcpyDeviceToDevice);

        // Get results
        cudaMemcpy(h_posX, d_posX, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_posY, d_posY, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_posZ, d_posZ, N * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_velX, d_velX, N * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_velY, d_velY, N * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_velZ, d_velZ, N * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; i++) {
            fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n",
                   i, step, h_posX[i], h_posY[i], h_posZ[i]);
        }
    }
    clock_t time_out = clock();
    double time_spent = (double)(time_out - time_in) / CLOCKS_PER_SEC;
    printf("time spent is %.6f\n", time_spent);

    fclose(fp);

    // Cleanup
    // Group device pointers
    double *d_ptrs[] = {d_mass, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ,
        d_accX_old, d_accY_old, d_accZ_old, d_accX_new, d_accY_new, d_accZ_new};
            
    // Group host pointers - fixed array to include only allocated pointers
    double *h_ptrs[] = {h_mass, h_posX, h_posY, h_posZ, h_velX, h_velY, h_velZ, com};

    // Free device memory
    for (int i = 0; i < sizeof(d_ptrs)/sizeof(d_ptrs[0]); i++) {
        cudaFree(d_ptrs[i]);
    }

    // Free host memory
    for (int i = 0; i < sizeof(h_ptrs)/sizeof(h_ptrs[0]); i++) {
        free(h_ptrs[i]);
    }

    return 0;
}