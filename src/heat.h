#ifndef HEAT_H
#define HEAT_H

/* Core C headers used across all builds (serial, OMP, CUDA, MPI). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Optional runtime headers enabled by Makefile feature flags. */
#ifdef USE_OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

/* Default runtime configuration (overridden from Makefile via -D flags). */
#ifndef NX_DEFAULT
#define NX_DEFAULT 4096
#endif
#ifndef NY_DEFAULT
#define NY_DEFAULT 4096
#endif
#ifndef NSTEPS_DEFAULT
#define NSTEPS_DEFAULT 10000
#endif
#ifndef SNAP_EVERY_DEFAULT
#define SNAP_EVERY_DEFAULT 100
#endif
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
#ifndef T_AMBIENT
#define T_AMBIENT 300.0f
#endif

/* Layer ids from bottom (PCB side) to top (die side). */
#define NUM_MATERIALS 4
typedef enum { MAT_FR4 = 0, MAT_COPPER = 1, MAT_TIM = 2, MAT_SILICON = 3 } material_t;

/* CUDA runtime guard used in all device calls (pattern from prior projects). */
#ifdef USE_CUDA
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
#endif

/* 2D row-major indexing helper: linear_index = j * nx + i. */
#define IDX(g, i, j) ((j) * (g)->nx + (i))

/* Shared simulation state used by CPU, CUDA, and MPI paths. */
typedef struct {
    int nx;              /* Global X dimension. */
    int ny;              /* Global Y dimension. */
    int local_ny;        /* Local Y rows owned by this rank. */
    int rank;            /* MPI rank id. */
    int nranks;          /* Number of MPI ranks. */
    float dx;            /* Cell spacing in X. */
    float dy;            /* Cell spacing in Y. */
    float dt;            /* Time step size. */
    float *T_old;        /* Host input temperature field. */
    float *T_new;        /* Host output temperature field. */
    float *d_T_old;      /* Device input temperature field. */
    float *d_T_new;      /* Device output temperature field. */
    int *material_id;    /* Material id per cell. */
} Grid;

/* Per-material property tables (defined in stencil_cpu.c). */
extern const float K_TABLE[NUM_MATERIALS];
extern const float RHOCP_TABLE[NUM_MATERIALS];
extern const float KY_TABLE[NUM_MATERIALS];

/* CPU stencil kernels (serial and OpenMP). */
void apply_stencil_serial(Grid *g);
void apply_stencil_omp(Grid *g);

/* CUDA allocation/launch/copy API (implemented in stencil_cuda.cu). */
#ifdef __cplusplus
extern "C" {
#endif
void cuda_alloc(Grid *g);
void cuda_free(Grid *g);
void launch_stencil_cuda(Grid *g);
void cuda_copy_to_host(Grid *g);
#ifdef __cplusplus
}
#endif

/* MPI halo exchange API (implemented in halo.c when present). */
void halo_alloc(Grid *g);
void halo_free(Grid *g);
void halo_exchange_cpu(Grid *g);
void halo_exchange_hybrid(Grid *g);

#endif
