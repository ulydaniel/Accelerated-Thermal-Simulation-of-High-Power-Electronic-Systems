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
                                     const float *rhocp_tbl, int nx, int ny,
                                     int local_ny, int global_y_start,
                                     float dx, float dy, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < 1 || i >= nx - 1 || j > local_ny) {
        return;
    }

    int m = material_id[j];
    int gy = global_y_start + (j - 1);
    float kx = kx_tbl[m];
    float ky = ky_tbl[m];
    float rhocp = rhocp_tbl[m];
    float ax = kx / rhocp;
    float ay = ky / rhocp;
    float qdot = 0.0f;
    if (m == MAT_SILICON) {
        float sigma = (float)nx * HOTSPOT_SIGMA_FRAC;
        float two_sigma2 = 2.0f * sigma * sigma;
        float cx0 = 0.25f * (float)nx;
        float cx1 = 0.50f * (float)nx;
        float cx2 = 0.75f * (float)nx;
        float cy = HOTSPOT_Y_FRAC * (float)ny;
        float dy0 = (float)gy - cy;
        float dx0 = (float)i - cx0;
        float dx1 = (float)i - cx1;
        float dx2 = (float)i - cx2;
        qdot = HOTSPOT_POWER_PEAK *
               (expf(-(dx0 * dx0 + dy0 * dy0) / two_sigma2) +
                expf(-(dx1 * dx1 + dy0 * dy0) / two_sigma2) +
                expf(-(dx2 * dx2 + dy0 * dy0) / two_sigma2));
    }

    size_t idx = (size_t)j * (size_t)nx + (size_t)i;
    float c = T_old[idx];
    float n = T_old[(size_t)(j + 1) * (size_t)nx + (size_t)i];
    float s = T_old[(size_t)(j - 1) * (size_t)nx + (size_t)i];
    float e = T_old[(size_t)j * (size_t)nx + (size_t)(i + 1)];
    float w = T_old[(size_t)j * (size_t)nx + (size_t)(i - 1)];

    T_new[idx] = c + dt * (ax * (e - 2.0f * c + w) / (dx * dx) +
                           ay * (n - 2.0f * c + s) / (dy * dy) +
                           qdot / rhocp);
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
                                     const float *rhocp_tbl, int nx, int ny,
                                     int local_ny, int global_y_start,
                                     float dx, float dy, float dt)
{
    extern __shared__ float tile[];
    int tile_pitch = blockDim.x + 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty + 1;
    int center = (ty + 1) * tile_pitch + (tx + 1);

    /* Center load for each thread. */
    if (i >= 0 && i < nx && j >= 0 && j <= local_ny + 1) {
        tile[center] = T_old[(size_t)j * (size_t)nx + (size_t)i];
    } else {
        tile[center] = T_AMBIENT;
    }

    /* Halo loads: each edge thread pulls one extra neighbor cell. */
    if (tx == 0) {
        if (i - 1 >= 0 && j >= 0 && j <= local_ny + 1) {
            tile[(ty + 1) * tile_pitch] =
                T_old[(size_t)j * (size_t)nx + (size_t)(i - 1)];
        } else {
            tile[(ty + 1) * tile_pitch] = T_AMBIENT;
        }
    }
    if (tx == blockDim.x - 1) {
        if (i + 1 < nx && j >= 0 && j <= local_ny + 1) {
            tile[(ty + 1) * tile_pitch + (blockDim.x + 1)] =
                T_old[(size_t)j * (size_t)nx + (size_t)(i + 1)];
        } else {
            tile[(ty + 1) * tile_pitch + (blockDim.x + 1)] = T_AMBIENT;
        }
    }
    if (ty == 0) {
        if (j - 1 >= 0 && i >= 0 && i < nx) {
            tile[tx + 1] = T_old[(size_t)(j - 1) * (size_t)nx + (size_t)i];
        } else {
            tile[tx + 1] = T_AMBIENT;
        }
    }
    if (ty == blockDim.y - 1) {
        if (j + 1 <= local_ny + 1 && i >= 0 && i < nx) {
            tile[(blockDim.y + 1) * tile_pitch + (tx + 1)] =
                T_old[(size_t)(j + 1) * (size_t)nx + (size_t)i];
        } else {
            tile[(blockDim.y + 1) * tile_pitch + (tx + 1)] = T_AMBIENT;
        }
    }

    /* Corner halo loads (only four corner threads do this). */
    if (tx == 0 && ty == 0) {
        if (i - 1 >= 0 && j - 1 >= 0) {
            tile[0] = T_old[(size_t)(j - 1) * (size_t)nx + (size_t)(i - 1)];
        } else {
            tile[0] = T_AMBIENT;
        }
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        if (i + 1 < nx && j - 1 >= 0) {
            tile[blockDim.x + 1] =
                T_old[(size_t)(j - 1) * (size_t)nx + (size_t)(i + 1)];
        } else {
            tile[blockDim.x + 1] = T_AMBIENT;
        }
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        if (i - 1 >= 0 && j + 1 <= local_ny + 1) {
            tile[(blockDim.y + 1) * tile_pitch] =
                T_old[(size_t)(j + 1) * (size_t)nx + (size_t)(i - 1)];
        } else {
            tile[(blockDim.y + 1) * tile_pitch] = T_AMBIENT;
        }
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        if (i + 1 < nx && j + 1 <= local_ny + 1) {
            tile[(blockDim.y + 1) * tile_pitch + (blockDim.x + 1)] =
                T_old[(size_t)(j + 1) * (size_t)nx + (size_t)(i + 1)];
        } else {
            tile[(blockDim.y + 1) * tile_pitch + (blockDim.x + 1)] = T_AMBIENT;
        }
    }

    __syncthreads();

    /* Compute only valid interior physical cells. */
    if (i < 1 || i >= nx - 1 || j > local_ny) {
        return;
    }

    int m = material_id[j];
    int gy = global_y_start + (j - 1);
    float kx = kx_tbl[m];
    float ky = ky_tbl[m];
    float rhocp = rhocp_tbl[m];
    float ax = kx / rhocp;
    float ay = ky / rhocp;
    float qdot = 0.0f;
    if (m == MAT_SILICON) {
        float sigma = (float)nx * HOTSPOT_SIGMA_FRAC;
        float two_sigma2 = 2.0f * sigma * sigma;
        float cx0 = 0.25f * (float)nx;
        float cx1 = 0.50f * (float)nx;
        float cx2 = 0.75f * (float)nx;
        float cy = HOTSPOT_Y_FRAC * (float)ny;
        float dy0 = (float)gy - cy;
        float dx0 = (float)i - cx0;
        float dx1 = (float)i - cx1;
        float dx2 = (float)i - cx2;
        qdot = HOTSPOT_POWER_PEAK *
               (expf(-(dx0 * dx0 + dy0 * dy0) / two_sigma2) +
                expf(-(dx1 * dx1 + dy0 * dy0) / two_sigma2) +
                expf(-(dx2 * dx2 + dy0 * dy0) / two_sigma2));
    }

    float c = tile[(ty + 1) * tile_pitch + (tx + 1)];
    float n = tile[(ty + 2) * tile_pitch + (tx + 1)];
    float s = tile[ty * tile_pitch + (tx + 1)];
    float e = tile[(ty + 1) * tile_pitch + (tx + 2)];
    float w = tile[(ty + 1) * tile_pitch + tx];

    T_new[(size_t)j * (size_t)nx + (size_t)i] =
        c + dt * (ax * (e - 2.0f * c + w) / (dx * dx) +
                  ay * (n - 2.0f * c + s) / (dy * dy) +
                  qdot / rhocp);
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
    dim3 block((unsigned int)g->block_x, (unsigned int)g->block_y);
    dim3 grid((g->nx + g->block_x - 1) / g->block_x,
              (g->local_ny + g->block_y - 1) / g->block_y);
    size_t shared_bytes = (size_t)(g->block_x + 2) * (size_t)(g->block_y + 2) * sizeof(float);

#ifdef USE_NAIVE_KERNEL
    stencil_kernel_naive<<<grid, block>>>(g->d_T_old, g->d_T_new, d_material_id,
                                          d_kx_tbl, d_ky_tbl, d_rhocp_tbl,
                                          g->nx, g->ny, g->local_ny,
                                          g->global_y_start, g->dx, g->dy, g->dt);
#else
    stencil_kernel_tiled<<<grid, block, shared_bytes>>>(g->d_T_old, g->d_T_new, d_material_id,
                                                         d_kx_tbl, d_ky_tbl, d_rhocp_tbl,
                                                         g->nx, g->ny, g->local_ny,
                                                         g->global_y_start, g->dx, g->dy, g->dt);
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
