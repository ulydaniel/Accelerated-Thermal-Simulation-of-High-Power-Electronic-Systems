#include "heat.h"

#ifdef USE_CUDA

/* Device-side copies of row materials + property tables used by kernels. */
static int *d_material_id = NULL;
static float *d_kx_tbl = NULL;
static float *d_ky_tbl = NULL;
static float *d_rhocp_tbl = NULL;

/*
 * Naive CUDA kernel:
 * - one thread updates one interior cell (i,j)
 * - j is offset by +1 so row 0 remains a ghost row
 * - left/right boundaries are skipped to preserve Dirichlet edges
 */
__global__ void stencil_kernel_naive(const float *T_old, float *T_new,
                                     const int *material_id,
                                     const float *kx_tbl, const float *ky_tbl,
                                     const float *rhocp_tbl, int nx, int local_ny,
                                     float dx, float dy, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < 1 || i >= nx - 1 || j > local_ny) {
        return;
    }

    int m = material_id[j];
    float kx = kx_tbl[m];
    float ky = ky_tbl[m];
    float rhocp = rhocp_tbl[m];
    float ax = kx / rhocp;
    float ay = ky / rhocp;

    int idx = j * nx + i;
    float c = T_old[idx];
    float n = T_old[(j + 1) * nx + i];
    float s = T_old[(j - 1) * nx + i];
    float e = T_old[j * nx + (i + 1)];
    float w = T_old[j * nx + (i - 1)];

    T_new[idx] = c + dt * (ax * (e - 2.0f * c + w) / (dx * dx) +
                           ay * (n - 2.0f * c + s) / (dy * dy));
}

/*
 * Tiled CUDA kernel:
 * - loads a TILE_WIDTH x TILE_WIDTH block plus 1-cell halo into shared memory
 * - computes the same stencil math as naive kernel
 * - reduces global memory traffic for neighbor reads
 */
__global__ void stencil_kernel_tiled(const float *T_old, float *T_new,
                                     const int *material_id,
                                     const float *kx_tbl, const float *ky_tbl,
                                     const float *rhocp_tbl, int nx, int local_ny,
                                     float dx, float dy, float dt)
{
    __shared__ float tile[TILE_WIDTH + 2][TILE_WIDTH + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty + 1;

    /* Center load for each thread. */
    if (i >= 0 && i < nx && j >= 0 && j <= local_ny + 1) {
        tile[ty + 1][tx + 1] = T_old[j * nx + i];
    } else {
        tile[ty + 1][tx + 1] = T_AMBIENT;
    }

    /* Halo loads: each edge thread pulls one extra neighbor cell. */
    if (tx == 0) {
        if (i - 1 >= 0 && j >= 0 && j <= local_ny + 1) {
            tile[ty + 1][0] = T_old[j * nx + (i - 1)];
        } else {
            tile[ty + 1][0] = T_AMBIENT;
        }
    }
    if (tx == TILE_WIDTH - 1) {
        if (i + 1 < nx && j >= 0 && j <= local_ny + 1) {
            tile[ty + 1][TILE_WIDTH + 1] = T_old[j * nx + (i + 1)];
        } else {
            tile[ty + 1][TILE_WIDTH + 1] = T_AMBIENT;
        }
    }
    if (ty == 0) {
        if (j - 1 >= 0 && i >= 0 && i < nx) {
            tile[0][tx + 1] = T_old[(j - 1) * nx + i];
        } else {
            tile[0][tx + 1] = T_AMBIENT;
        }
    }
    if (ty == TILE_WIDTH - 1) {
        if (j + 1 <= local_ny + 1 && i >= 0 && i < nx) {
            tile[TILE_WIDTH + 1][tx + 1] = T_old[(j + 1) * nx + i];
        } else {
            tile[TILE_WIDTH + 1][tx + 1] = T_AMBIENT;
        }
    }

    /* Corner halo loads (only four corner threads do this). */
    if (tx == 0 && ty == 0) {
        if (i - 1 >= 0 && j - 1 >= 0) {
            tile[0][0] = T_old[(j - 1) * nx + (i - 1)];
        } else {
            tile[0][0] = T_AMBIENT;
        }
    }
    if (tx == TILE_WIDTH - 1 && ty == 0) {
        if (i + 1 < nx && j - 1 >= 0) {
            tile[0][TILE_WIDTH + 1] = T_old[(j - 1) * nx + (i + 1)];
        } else {
            tile[0][TILE_WIDTH + 1] = T_AMBIENT;
        }
    }
    if (tx == 0 && ty == TILE_WIDTH - 1) {
        if (i - 1 >= 0 && j + 1 <= local_ny + 1) {
            tile[TILE_WIDTH + 1][0] = T_old[(j + 1) * nx + (i - 1)];
        } else {
            tile[TILE_WIDTH + 1][0] = T_AMBIENT;
        }
    }
    if (tx == TILE_WIDTH - 1 && ty == TILE_WIDTH - 1) {
        if (i + 1 < nx && j + 1 <= local_ny + 1) {
            tile[TILE_WIDTH + 1][TILE_WIDTH + 1] = T_old[(j + 1) * nx + (i + 1)];
        } else {
            tile[TILE_WIDTH + 1][TILE_WIDTH + 1] = T_AMBIENT;
        }
    }

    __syncthreads();

    /* Compute only valid interior physical cells. */
    if (i < 1 || i >= nx - 1 || j > local_ny) {
        return;
    }

    int m = material_id[j];
    float kx = kx_tbl[m];
    float ky = ky_tbl[m];
    float rhocp = rhocp_tbl[m];
    float ax = kx / rhocp;
    float ay = ky / rhocp;

    float c = tile[ty + 1][tx + 1];
    float n = tile[ty + 2][tx + 1];
    float s = tile[ty][tx + 1];
    float e = tile[ty + 1][tx + 2];
    float w = tile[ty + 1][tx];

    T_new[j * nx + i] = c + dt * (ax * (e - 2.0f * c + w) / (dx * dx) +
                                  ay * (n - 2.0f * c + s) / (dy * dy));
}

/* Allocate and upload all GPU-side buffers used during time stepping. */
extern "C" void cuda_alloc(Grid *g)
{
    size_t bytes = (size_t)g->nx * (size_t)(g->local_ny + 2) * sizeof(float);
    size_t mat_bytes = (size_t)(g->local_ny + 2) * sizeof(int);
    size_t tbl_bytes = (size_t)NUM_MATERIALS * sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)&g->d_T_old, bytes));
    CUDA_CHECK(cudaMalloc((void **)&g->d_T_new, bytes));
    CUDA_CHECK(cudaMemcpy(g->d_T_old, g->T_old, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_material_id, mat_bytes));
    CUDA_CHECK(cudaMemcpy(d_material_id, g->material_id, mat_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_kx_tbl, tbl_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ky_tbl, tbl_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_rhocp_tbl, tbl_bytes));
    CUDA_CHECK(cudaMemcpy(d_kx_tbl, K_TABLE, tbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ky_tbl, KY_TABLE, tbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhocp_tbl, RHOCP_TABLE, tbl_bytes, cudaMemcpyHostToDevice));
}

/* Free all GPU allocations owned by this translation unit. */
extern "C" void cuda_free(Grid *g)
{
    if (g->d_T_old != NULL) {
        CUDA_CHECK(cudaFree(g->d_T_old));
        g->d_T_old = NULL;
    }
    if (g->d_T_new != NULL) {
        CUDA_CHECK(cudaFree(g->d_T_new));
        g->d_T_new = NULL;
    }
    if (d_material_id != NULL) {
        CUDA_CHECK(cudaFree(d_material_id));
        d_material_id = NULL;
    }
    if (d_kx_tbl != NULL) {
        CUDA_CHECK(cudaFree(d_kx_tbl));
        d_kx_tbl = NULL;
    }
    if (d_ky_tbl != NULL) {
        CUDA_CHECK(cudaFree(d_ky_tbl));
        d_ky_tbl = NULL;
    }
    if (d_rhocp_tbl != NULL) {
        CUDA_CHECK(cudaFree(d_rhocp_tbl));
        d_rhocp_tbl = NULL;
    }
}

/* Launch one stencil step on GPU (sync handled by caller in main loop). */
extern "C" void launch_stencil_cuda(Grid *g)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((g->nx + TILE_WIDTH - 1) / TILE_WIDTH,
              (g->local_ny + TILE_WIDTH - 1) / TILE_WIDTH);

#ifdef USE_NAIVE_KERNEL
    stencil_kernel_naive<<<grid, block>>>(g->d_T_old, g->d_T_new, d_material_id,
                                          d_kx_tbl, d_ky_tbl, d_rhocp_tbl,
                                          g->nx, g->local_ny, g->dx, g->dy, g->dt);
#else
    stencil_kernel_tiled<<<grid, block>>>(g->d_T_old, g->d_T_new, d_material_id,
                                          d_kx_tbl, d_ky_tbl, d_rhocp_tbl,
                                          g->nx, g->local_ny, g->dx, g->dy, g->dt);
#endif
    CUDA_CHECK(cudaGetLastError());
}

/* Copy latest device field back to host for snapshot output. */
extern "C" void cuda_copy_to_host(Grid *g)
{
    size_t bytes = (size_t)g->nx * (size_t)(g->local_ny + 2) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(g->T_old, g->d_T_old, bytes, cudaMemcpyDeviceToHost));
}

#endif
