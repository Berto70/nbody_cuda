#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <memory.h>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <chrono>

//##############################################################################
//# CONSTANTS
//##############################################################################
#define N 16384 // number of particles 
#define G 1.0 // gravitational constant

#define evolt 0.315f
#define DT 0.000001f // time step 
#define EPS2 1e-9f // softening parameter
#define STEPS 1000 // simulation steps
#define L 100.0 // box size

//##############################################################################
//# CPU FUNCTIONS
//##############################################################################

/**
 * @brief Compute gravitational acceleration between two bodies using Newton's law
 * 
 * This function calculates the gravitational acceleration exerted by body j
 * on body i, and adds it to the current acceleration of body i. It implements
 * the pairwise interaction for an N-body gravitational simulation.
 * 
 * The calculation includes a softening factor (EPS2) to prevent numerical 
 * instabilities when bodies are very close to each other.
 * 

 * FLOP count: 20 floating-point operations per call.
 * 
 * @param bi Position and mass of body i (x, y, z, mass) - the current body
 * @param bj Position and mass of body j (x, y, z, mass) - the interacting body
 * @param ai Current acceleration vector of body i
 * @return Updated acceleration vector of body i after adding contribution from body j
 */
__host__ float3 computeAcceleration(float4 bi, float4 bj, float3 ai) {
    float3 r;

    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    
    // ||r_ij||^2 + eps^2 [6 FLOPS]
    float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + EPS2;

    // 1/distSqr^(3/2) [4 FLOPS]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f / sqrtf(distSixth);
    
    // m_j * 1/distSqr^(3/2) [1 FLOP]
    float s = bj.w * invDistCube;
    
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += s * r.x;
    ai.y += s * r.y;
    ai.z += s * r.z;
    
    return ai; // tot [20 FLOPS]
}

/**
 * @brief Calculates the acceleration for all particles in the simulation
 *
 * This function computes the gravitational forces between all pairs of particles
 * and calculates the resulting acceleration for each particle.
 *
 * @param[in] pos Array of particle positions and masses
 * @param[out] acc Array for storing the calculated accelerations
 */
__host__ void forceCalculation(float4 *pos, float4 *acc) { 
    for (int i = 0; i < N; i++) {
        float3 accel = {0.0f, 0.0f, 0.0f};
        float4 myPos = pos[i];
        
        for (int j = 0; j < N; j++) {
            accel = computeAcceleration(myPos, pos[j], accel);
        }
        
        // Save the result
        acc[i].x = accel.x;
        acc[i].y = accel.y;
        acc[i].z = accel.z;
        acc[i].w = 0.0f;
    }
}

__host__ void forceCalculationSymmetry(float4 *pos, float4 *acc) { 
    for (int i = 0; i < N-1; i++) {
        float3 accel = {0.0f, 0.0f, 0.0f};
        float4 myPos = pos[i];
        
        for (int j = i+1; j < N; j++) {
            accel = computeAcceleration(myPos, pos[j], accel);
        }
        
        // Save the result
        acc[i].x = accel.x;
        acc[i].y = accel.y;
        acc[i].z = accel.z;
        acc[i].w = 0.0f;
    }
}

/**
 * @brief Updates positions of particles in an N-body simulation
 *
 * This function updates the positions of particles based on their velocities and
 * accelerations using the Verlet integration method.
 *
 * @param[in] n Number of bodies
 * @param pos Array of particle positions (and mass) as float4 where x,y,z are positions
 * @param[in] vel Array of particle velocities as float4 where x,y,z are velocity components
 * @param[in] acc Array of particle accelerations as float4 where x,y,z are acceleration components
 * @param[in] dt Time step for the simulation
 */
__host__ void updatePositions(int n, float4 *pos, float4 *vel, float4 *acc, float dt) { 
    for (int i = 0; i < n; i++) { 
        pos[i].x += vel[i].x*dt + 0.5 * acc[i].x * dt * dt; 
        pos[i].y += vel[i].y*dt + 0.5 * acc[i].y * dt * dt; 
        pos[i].z += vel[i].z*dt + 0.5 * acc[i].z * dt * dt;  
    } 
}

/**
 * @brief Updates velocities of particles in an N-body simulation
 *
 * This function updates the velocity of each body based on old and new accelerations
 * using the velocity Verlet integration method.
 *
 * @param[in] n Number of bodies
 * @param vel Array of float4 values representing velocities (x,y,z components)
 * @param[in] oldAcc Array of float4 values representing accelerations at time t
 * @param[in] newAcc Array of float4 values representing accelerations at time t+dt
 * @param[in] dt Time step size
 */
__host__ void updateVelocities(int n, float4 *vel, float4 *oldAcc, float4 *newAcc, double dt) { 
    for (int i = 0; i < n; i++) { 
        vel[i].x += 0.5 * (oldAcc[i].x + newAcc[i].x) * dt; 
        vel[i].y += 0.5 * (oldAcc[i].y + newAcc[i].y) * dt; 
        vel[i].z += 0.5 * (oldAcc[i].z + newAcc[i].z) * dt; 
    } 
}

/**
 * @brief Computes the total energy of an N-body system
 * 
 * This function calculates both kinetic and potential energy for each particle
 * and accumulates them to get the total energy of the system.
 * 
 * @param[in] n Number of bodies
 * @param[in] pos Array of float4 values where (x,y,z) is the position and w is the mass of each particle
 * @param[in] vel Array of float4 values where (x,y,z) is the velocity of each particle
 * @return The total energy of the system
 */
float computeEnergy(int n, float4 *pos, float4 *vel) {
    float kinetic = 0.0f;
    float potential = 0.0f;
    
    for (int i = 0; i < n; i++) {
        // Calculate potential energy
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float3 r;
                r.x = pos[j].x - pos[i].x;
                r.y = pos[j].y - pos[i].y;
                r.z = pos[j].z - pos[i].z;
                
                float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
                potential -= (pos[i].w * pos[j].w) / sqrt(distSqr);
            }
        }
        
        // Calculate kinetic energy
        kinetic += 0.5 * pos[i].w * (vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z);
    }
    
    return kinetic + potential;
}

/**
 * @brief Computes the center of mass for a system of bodies
 *
 * This function calculates the center of mass for a system of N bodies by computing
 * the weighted average of positions based on the mass of each body. 
 *
 * @param[in] pos Array of float4 values where x,y,z represent positions and w represents mass
 * @param[out] com Pointer to float3 where the calculated center of mass will be stored
 */
void compute_com(float4 *pos, float3 *com) {
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
 */
void ic_random_uniform(int n_particles, 
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
 * @brief Evolves an N-body system for a specified number of time steps using CPU calculations
 * 
 * This function initializes the N-body system, runs the simulation for a specified
 * number of steps, and handles data saving if requested. It's a CPU-based version
 * adapted from a CUDA implementation.
 *
 * @param[in] sims Simulation scenario number (2=two-body, 3=three-body, >3=random)
 * @param[in] save_data If non-zero, save particle positions to a file
 * @param[in] energy If non-zero, compute and save energy data
 * @param[in] save_steps Interval for saving data (save every save_steps iterations)
 * @return 0 if successful, 1 if error occurred
 */
int evolveSystem(int sims, int save_data, int energy, int save_steps) { 
    // Host memory
    float4 *h_pos = (float4*)malloc(N * sizeof(float4));
    float4 *h_vel = (float4*)malloc(N * sizeof(float4));
    float4 *h_acc_old = (float4*)malloc(N * sizeof(float4));
    float4 *h_acc_new = (float4*)malloc(N * sizeof(float4));
    float3 *com = (float3*)malloc(sizeof(float3));

    FILE *fp = NULL;
    FILE *fpe = NULL;
    float total_energy = 0.0f;

    // Initialize data
    if (sims > 3) {
        double mass_range[2] = {1.0, 10.0};      // Masses between 1.0 and 10.0
        double pos_range[2] = {-L/2, L/2};     // Positions between -50.0 and 50.0 --> L = 100.0
        double vel_range[2] = {-1.0, 1.0};       // Velocities between -1.0 and 1.0

        ic_random_uniform(N, mass_range, pos_range, vel_range, h_pos, h_vel, 1);
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

    // Initialize old acceleration to zero
    for (int i = 0; i < N; i++) {
        h_acc_old[i].x = 0.0f;
        h_acc_old[i].y = 0.0f;
        h_acc_old[i].z = 0.0f;
        h_acc_old[i].w = 0.0f;
    }

    if (save_data) {
        fp = fopen("serial_output.csv", "w");
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
        fpe = fopen("serial_energy.csv", "w");
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

    // Run simulation
    for (int step = 0; step < STEPS; step++) {
        // Calculate forces and update accelerations
        forceCalculationSymmetry(h_pos, h_acc_new);
        
        // Update positions
        updatePositions(N, h_pos, h_vel, h_acc_old, DT);
        
        // Update velocities
        updateVelocities(N, h_vel, h_acc_old, h_acc_new, DT);

        // Compute total energy
        if (energy) {
            total_energy = computeEnergy(N, h_pos, h_vel);
        }
        
        // Copy new acceleration to old acceleration
        for (int i = 0; i < N; i++) {
            h_acc_old[i] = h_acc_new[i];
        }
        
        if (save_data && (step % save_steps == 0)) {
            if (energy) {
                fprintf(fpe, "%d,%.6f\n", step, total_energy);
            }   
            
            for (int i = 0; i < N; i++) {
                fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n",
                    i, step, h_pos[i].x, h_pos[i].y, h_pos[i].z);
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
    free(h_pos);
    free(h_vel);
    free(h_acc_old);
    free(h_acc_new);
    free(com);
    
    return 0;
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
 */
void computePerfStats(double &interactionsPerSecond, double &gflops, 
    float milliseconds) {
const int flopsPerInteraction = 20;
interactionsPerSecond = (float)N * (float)N;
interactionsPerSecond *= 1e-9 * STEPS * 1000 / milliseconds;
gflops = interactionsPerSecond * (float)flopsPerInteraction;

}

void runBenchmark() {
        
    // once without timing to prime the CPU
    evolveSystem(N,0,0,100);

    clock_t t_start = clock();  
    evolveSystem(N,0,0,100);
    clock_t t_end = clock();  

    float milliseconds = ((float)t_end - (float)t_start) / CLOCKS_PER_SEC * 1000;
    double interactionsPerSecond = 0;
    double gflops = 0;
    computePerfStats(interactionsPerSecond, gflops, milliseconds);

    FILE *fp = fopen("nbody_cuda/data/serial_gflops.csv", "a");
    if (!fp) {
        printf("Error opening file\n");
        return;
    }
    // If the file is empty, write the header.
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "n_bodies,gflopd,inter\n");
    }
    fprintf(fp, "%d,%0.3f,%0.3f\n", N, gflops, interactionsPerSecond);
    fclose(fp);
    
    //printf("%d bodies, total time for %d iterations: %0.3f ms\n", 
           //N, STEPS, milliseconds);
    // printf("= %0.3f billion interactions per second\n", interactionsPerSecond);
    // printf("= %0.3f GFLOP/s at %d flops per interaction\n\n", gflops, 20);
    
}

int main() {
    // Run the benchmark
     
        runBenchmark();

    // Run the simulation with data saving if needed
    // int save_data = 1;
    // int energy = 1;
    // int save_steps = 10;
    // evolveSystem(N, save_data, energy, save_steps);
    
    return 0;
}

