// System Utilities
#include <iostream>
#include <cmath>
#include <assert.h>
#include <helper_string.h>

// CUDA Runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS helpers
#include <helper_functions.h>
#include <helper_cuda.h>

// Define Block Size
#define BLOCK_SIZE 32


// Returns a matrix filled with garbage values
float * get_matrix(int dim) {

    // compute required memory for matrix
    int size = dim*dim*sizeof(float);
    
    // allocate memory for matrix
    float * matrix = (float *) malloc(size);

    return matrix;
}

// Returns matrix filled with random float values
// Matrix formatted in row major order
float * randomize_matrix(int dim) {

    // initialize empty matrix
    float * matrix = get_matrix(dim);

    // fill matrix with random floats
    for (int i = 0; i < dim*dim; i++) {

        // fill with values in the range [0.0, 1.0] to check decimal accuracy
        matrix[i] = rand()/(float)RAND_MAX;
    }

    return matrix;
}

// Sets up Device Memory for matrices
float * device_matrix(int dim) {

    // initialize pointer
    void * dev_ptr;

    // compute required memory for matrix
    int size = dim*dim*sizeof(float);

    // allocate device memory
    checkCudaErrors(cudaMalloc(&dev_ptr, size));

    return (float *) dev_ptr;
}

// Copies matrix from host memory to device memory
void host_to_device(const void* host_ptr, void * dev_ptr, int dim) {

    // compute memory to be copied
    int size = dim*dim*sizeof(float);

    // copy memory to device
    checkCudaErrors(cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice));

}

// Copies matrix from device memory to host memory
void device_to_host(const void * dev_ptr, void * host_ptr, int dim) {

    // compute memory to be copied
    int size = dim*dim*sizeof(float);

    // copy memory to host
    checkCudaErrors(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));

}

// Compares the result of our implementation of matmul to cublas
// Ensures difference between two entries is less than err
void compare_matrix(float * C_alg, float * C_cub, int size, float err) {

    // number of entries in matrix
    int num_entries = size * size;


    // check every entry
    for (int i = 0; i < num_entries; i++) {
        
        // error outside of acceptable range
        if (std::fabs(C_alg[i] - C_cub[i]) > err) {
            std::cout << "Failed: entry " << i <<" does not match" << std::endl;
            std::cout << "Alg entry: " << C_alg[i] << ", Cublas entry: " << C_cub[i] << std::endl;
            std::cout << "Difference: " << std::fabs(C_alg[i] - C_cub[i]) << std::endl;
            return;
        }
    }

    // matrices are identical to within the error margin
    std::cout << "Passed" << std::endl;
}

// Shared Memory Cache Blocking Matrix Multiplication Algorithm
// Assumes size is a multiple of BLOCK_SIZE
__global__ void mat_mul(const float * A, const float * B, float * C, int size) {

    // Initialize Scratch Space
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    // Compute col index of result in C
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    // Compute row index of result in C
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    // Compute in memory index
    int index = row*size + col;

    float total = 0.0;


    for (int i = 0; i < size; i += BLOCK_SIZE) {

        // Each thread is responsible for loading a float
        // from global memory to shared memory


        // Index of global mem that thread loads from
        int A_index_global = row * size + threadIdx.x + i;
        int B_index_global = (i+threadIdx.y) * size + col;

        
        // Load into shared memory
        A_shared[threadIdx.y][threadIdx.x] = A[A_index_global];
        B_shared[threadIdx.y][threadIdx.x] = B[B_index_global];

        // Let threads load data into shared mem        
        __syncthreads();
        
        
        // Compute partial totals
        for (int j = 0; j < BLOCK_SIZE; j++) {

            total += A_shared[threadIdx.y][j] *
            B_shared[j][threadIdx.x];

        }

        // Let threads finish computing before loading next block
        __syncthreads();
    }

    // Store the result in device memory
    C[index] = total;
}




// Runs test comparing implementation of matmul to cublas
// for two square matrices of dim size x size
void run_test(int size) {

    // create matrices A and B and fill with random values
    float * A = randomize_matrix(size);
    float * B = randomize_matrix(size);

    // scratch space to hold end result of matmul
    float * C_cub = get_matrix(size);
    float * C_alg = get_matrix(size);

    // allocate device memory
    float * A_d = device_matrix(size);
    float * B_d = device_matrix(size);
    float * C_d_cub = device_matrix(size);
    float * C_d_alg = device_matrix(size);

    // set parameters
    float alpha = 1.0f;
    float beta = 0.0f;

    // set two dimensional grid, enough to cover matrix
    dim3 grid_dim(size/BLOCK_SIZE, size/BLOCK_SIZE, 1); 

    // set two dimensional thread block of 1024 threads
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    // initialize cublas handle
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

    // Copy A from host memory to device memory
    host_to_device(A, A_d, size);

    // Copy B from host memory to device memory
    host_to_device(B, B_d, size);

    // Initialize events for timing diagnostics'
    cudaEvent_t start_cub;
    cudaEvent_t stop_cub;
    cudaEvent_t start_alg;
    cudaEvent_t stop_alg;

    // Create CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start_cub));
    checkCudaErrors(cudaEventCreate(&stop_cub));
    checkCudaErrors(cudaEventCreate(&start_alg));
    checkCudaErrors(cudaEventCreate(&stop_alg));

    // Warmup Routine
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, size, size, size, &alpha, A_d, size, B_d, size, &beta, C_d_cub, size));

    // Let Warmup Routine Finish
    checkCudaErrors(cudaDeviceSynchronize());

    // Record start point
    checkCudaErrors(cudaEventRecord(start_cub, NULL));

    // Perform matrix multiplication
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, B_d, size, A_d, size, &beta, C_d_cub, size));

    // Record end point
    checkCudaErrors(cudaEventRecord(stop_cub, NULL));

    // Wait until the cuda event terminates
    checkCudaErrors(cudaEventSynchronize(stop_cub));

    // Record elapsed time
    float ms_elapsed_cub;
    checkCudaErrors(cudaEventElapsedTime(&ms_elapsed_cub, start_cub, stop_cub));

    // floating point operations per matmul
    double flops_cub = 2.0 * ((double) size) * ((double) size) * ((double) size);
    double gf_per_s_cub = flops_cub * 1.0e-9f /(ms_elapsed_cub / 1000.0f);

    // Print diagnostics
    printf("Cublas Performance Metrics: \n %.2f GFlops/s \n %.3f ms\n", gf_per_s_cub, ms_elapsed_cub);

    // Record start point
    checkCudaErrors(cudaEventRecord(start_alg, NULL));

    // Run implemented algorithm
    mat_mul<<<grid_dim, block_dim>>>(A_d, B_d, C_d_alg, size);

    // Allow all threads to finish
    cudaDeviceSynchronize();

    // Record stop point
    checkCudaErrors(cudaEventRecord(stop_alg, NULL));

    // Wait until the cuda event terminates
    checkCudaErrors(cudaEventSynchronize(stop_alg));

    // Record elapsed time
    float ms_elapsed_alg;
    checkCudaErrors(cudaEventElapsedTime(&ms_elapsed_alg, start_alg, stop_alg));

    // floating point operations per matmul
    double flops_alg = 2.0 * ((double) size) * ((double) size) * ((double) size);
    double gf_per_s_alg = flops_alg * 1.0e-9f /(ms_elapsed_alg / 1000.0f);

    // Print diagnostics
    printf("Shared Memory Cache Blocking Algorithm Performance Metrics: \n %.2f GFlops/s \n %.3f ms\n", gf_per_s_alg, ms_elapsed_alg);

    // Copy reuslt back to host memory
    device_to_host(C_d_cub, C_cub, size);
    device_to_host(C_d_alg, C_alg, size);

    // Check for errors
    compare_matrix(C_alg, C_cub, size, 1.0e-1f);

    // cleanup memory
    free(A);
    free(B);
    free(C_cub);
    free(C_alg);

    // cleanup events
    cudaEventDestroy(start_cub);
    cudaEventDestroy(stop_cub);
    cudaEventDestroy(start_alg);
    cudaEventDestroy(stop_alg);

    // cleanup device memory
    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d_cub));
    checkCudaErrors(cudaFree(C_d_alg));

    // destroy cublas handle
    checkCudaErrors(cublasDestroy(handle));

}



int main(int argc, char * argv[]) {

    int size = std::stoi(argv[1]);
    run_test(size);
    return 0;
}