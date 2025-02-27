#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h>
#include <time.h>

#define N 20 // number of particles 
#define BLOCK_SIZE 32 
#define G 1.0 // gravitational constant
#define DT 0.0001 // time step 
#define EPSILON 0.00001 // softening parameter
#define STEPS 100000 // simulation steps
#define L 100.0 // box size

/* Updating acceleration of one body i given its interaction
with another body j */
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r;
    // r_ij  [3 FLOPS] --> calculate the distance between two bodies
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS] 
    // --> calculate the distance squared between two bodies
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPSILON*EPSILON;
    // invDistCube = 1/dist^(3/2)  [3 FLOPS (1 mul, 1 sqrt, 1 inv)] 
    // --> calculate the inverse distance cube
    float invDistCube = 1.0f / (sqrtf(distSqr) * distSqr);

    // s = m_j * invDistCube  [1 FLOP]
    // --> calculate the force of the interaction
    float s = bj.w * invDistCube;

    // a_i = a_i + s * r_ij  [6 FLOPS]
    // --> calculate the acceleration of the body
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

/* Interaction in a n_threadsXn_threads tile */
__device__ float3 tileCalculation(float4 myPos, float3 acc) {
    
    extern __shared__ float4 sharedPos[];

    for (int i = 0; i < blockDim.x; i++) {
        acc = bodyBodyInteraction(myPos, sharedPos[i], acc);
    }

    return acc;
}

/* Kernel executed by a block with n_threads to compute the acc
for n_threads bodies as a result of all N interactions */
__global__ void forceCalculation(void *d_pos, void *d_acc) {
    extern __shared__ float4 sharedPos[];
    float4 *pos = (float4 *)d_pos;
    float4 *acc = (float4 *)d_acc;
    float4 myPos;
    int i, tile;

    float3 myAcc = {0.0f, 0.0f, 0.0f};
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    myPos = pos[gidx];
    
    for (i = 0, tile = 0; i < N; i += blockDim.x, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        sharedPos[threadIdx.x] = pos[idx];
        __syncthreads();

        myAcc = tileCalculation(myPos, myAcc);
        __syncthreads();
    }
    // save result in global memory for integration step

    float4 acc4 = {myAcc.x, myAcc.y, myAcc.z, 0.0f};

    acc[gidx] = acc4;
}

/* Integration step - Leapfrog */

__global__ void integrateBodies(float4* newPos, float4* newVel,
                                float4* oldPos, float4* oldVel,
                                float4* acceleration, float tstep) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float4 pos = oldPos[idx];
        float4 acc = acceleration[idx]; // Use pre-calculated acceleration
        float4 vel = oldVel[idx];

        // update position
        pos.x += vel.x * tstep + 0.5f * acc.x * tstep * tstep;
        pos.y += vel.y * tstep + 0.5f * acc.y * tstep * tstep;
        pos.z += vel.z * tstep + 0.5f * acc.z * tstep * tstep;

        // update velocity
        vel.x += acc.x * tstep;
        vel.y += acc.y * tstep;
        vel.z += acc.z * tstep;

        newPos[idx] = pos;
        newVel[idx] = vel;
    }                                
}

/* void allocateNBodyArrays(float* vel[2], int N)
{
    // 4 floats each for alignment reasons
    unsigned int memSize = sizeof( float) * 4 * N;
    
    cudaMalloc((void**)&vel[0], memSize);
    cudaMalloc((void**)&vel[1], memSize);
}

void deleteNBodyArrays(float* vel[2])
{
    cudaFree(vel[0]);
    cudaFree(vel[1]);
}

void copyArrayFromDevice(float* host, const float* device, unsigned int pbo, int N)
{   
    cudaMemcpy(host, device, N*4*sizeof(float), cudaMemcpyDeviceToHost);
}

void copyArrayToDevice(float* device, const float* host, int N)
{
    cudaMemcpy(device, host, N*4*sizeof(float), cudaMemcpyHostToDevice);
} */

void integrateNbodySystem(float* newPos, float* newVel, 
                     float* oldPos, float* oldVel, float tstep, int numBodies) {

    int sharedMemSize = BLOCK_SIZE * 1 * sizeof(float4); // 4 floats for pos
    
    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 grid((numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    // Allocate memory for acceleration
    float4* d_acc;
    cudaMalloc((void**)&d_acc, numBodies * sizeof(float4));

    // Calculate forces first
    forceCalculation<<< grid, threads, sharedMemSize >>>((float4*)oldPos, d_acc);

    // Then integrate bodies with calculated acceleration
    integrateBodies<<< grid, threads, sharedMemSize >>>((float4*)newPos, (float4*)newVel,
                                                       (float4*)oldPos, (float4*)oldVel,
                                                       d_acc, tstep);
    
    // Free acceleration memory
    cudaFree(d_acc);
}

void compute_com(float4* bodies, float* com, int N) {
    float total_mass = 0.0f;
    com[0] = com[1] = com[2] = 0.0f;
    
    for (int i = 0; i < N; i++) {
        total_mass += bodies[i].w;  // Mass is stored in w component
        com[0] += bodies[i].w * bodies[i].x;
        com[1] += bodies[i].w * bodies[i].y;
        com[2] += bodies[i].w * bodies[i].z;
    }
    
    com[0] /= total_mass;
    com[1] /= total_mass;
    com[2] /= total_mass;
}

void init_random_uniform( float4* h_pos, float4* h_vel, int N) {
    
    // Initialize random number generator
    srand(42);
    
    // Mass and position/velocity ranges
    float mass_range[2] = {0.1f, 1.0f};  // Range for particle masses
    float pos_range[2] = {-L/2, L/2};    // Range for positions (box centered at origin)
    float vel_range[2] = {-1.0f, 1.0f};  // Range for initial velocities
    
    // Generate uniform random values within ranges
    for (int i = 0; i < N; i++) {
        // Position and mass (x,y,z,w where w = mass)
        h_pos[i].x = pos_range[0] + (rand() / (float)RAND_MAX) * (pos_range[1] - pos_range[0]);
        h_pos[i].y = pos_range[0] + (rand() / (float)RAND_MAX) * (pos_range[1] - pos_range[0]);
        h_pos[i].z = pos_range[0] + (rand() / (float)RAND_MAX) * (pos_range[1] - pos_range[0]);
        h_pos[i].w = mass_range[0] + (rand() / (float)RAND_MAX) * (mass_range[1] - mass_range[0]);
        
        // Velocity (x,y,z,w where w = 0)
        h_vel[i].x = vel_range[0] + (rand() / (float)RAND_MAX) * (vel_range[1] - vel_range[0]);
        h_vel[i].y = vel_range[0] + (rand() / (float)RAND_MAX) * (vel_range[1] - vel_range[0]);
        h_vel[i].z = vel_range[0] + (rand() / (float)RAND_MAX) * (vel_range[1] - vel_range[0]);
        h_vel[i].w = 0.0f; // Not used for velocity
    }

    // Center the system by subtracting the CoM
    float com[3];
    compute_com(h_pos, com, N);
    
    // Shift positions to center of mass frame
    for (int i = 0; i < N; i++) {
        h_pos[i].x -= com[0];
        h_pos[i].y -= com[1];
        h_pos[i].z -= com[2];
    }
}

void init() {
    // Host memory for particles
    float4 *h_pos = (float4*)malloc(N * sizeof(float4));
    float4 *h_vel = (float4*)malloc(N * sizeof(float4));
    
    // Device memory for positions and velocities (double buffering)
    float4 *d_pos[2];
    float4 *d_vel[2];
    
    // Allocate device memory
    unsigned int memSize = sizeof(float4) * N;
    cudaMalloc((void**)&d_pos[0], memSize);
    cudaMalloc((void**)&d_pos[1], memSize);
    cudaMalloc((void**)&d_vel[0], memSize);
    cudaMalloc((void**)&d_vel[1], memSize);

    // Initialize positions and velocities
    init_random_uniform(h_pos, h_vel, N);
    
    // Copy data from host to device
    cudaMemcpy(d_pos[0], h_pos, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel[0], h_vel, memSize, cudaMemcpyHostToDevice);
    
    // Free host memory
    free(h_pos);
    free(h_vel);
    
    // Run simulation
    int readBuf = 0;    // Read buffer index
    int writeBuf = 1;   // Write buffer index
    
    // Main simulation loop
    clock_t start = clock();
    for (int step = 0; step < STEPS; step++) {
        // Launch kernel to calculate forces and integrate
        integrateNbodySystem((float*)d_pos[writeBuf], (float*)d_vel[writeBuf], 
                          (float*)d_pos[readBuf], (float*)d_vel[readBuf],
                          DT, N);
        
        // Swap buffers
        /* The double-buffering approach (using `readBuf` and `writeBuf`) 
        operates at a higher level - it's for synchronization 
        **between kernel launches** across simulation time steps. */
        int temp = readBuf;
        readBuf = writeBuf;
        writeBuf = temp;
    }
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %f\n", time);
    // Copy final positions back to host if needed
    // (This would be here if you want to save results)
    
    // Free device memory
    cudaFree(d_pos[0]);
    cudaFree(d_pos[1]);
    cudaFree(d_vel[0]);
    cudaFree(d_vel[1]);
}

int main() {
    init();
    return 0;
}



