#include "heat.h"
#include <errno.h>
#include <sys/stat.h>
#include <time.h>

// === Section 1: Static helpers ===

/*
 * Parse command-line overrides for grid size, time steps, and output settings.
 * We keep defaults in heat.h so Makefile flags and CLI flags can both control runs.
 */
static void parse_args(int argc, char **argv, int *nx, int *ny, int *nsteps,
                       int *snap_every, char **outdir)
{
    *nx = NX_DEFAULT;
    *ny = NY_DEFAULT;
    *nsteps = NSTEPS_DEFAULT;
    *snap_every = SNAP_EVERY_DEFAULT;
    *outdir = (char *)"results";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            *nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ny") == 0 && i + 1 < argc) {
            *ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            *nsteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--snap-every") == 0 && i + 1 < argc) {
            *snap_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--outdir") == 0 && i + 1 < argc) {
            *outdir = argv[++i];
        } else {
            fprintf(stderr,
                    "Usage: %s [--nx N] [--ny N] [--steps N] [--snap-every N] [--outdir PATH]\n",
                    argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (*nx < 3 || *ny < 1 || *nsteps < 1 || *snap_every < 1) {
        fprintf(stderr, "Invalid args: nx>=3 ny>=1 steps>=1 snap-every>=1 required.\n");
        exit(EXIT_FAILURE);
    }
}

static float compute_dt(float dx, float dy)
{
    /* Conservative CFL: use the largest diffusivity across all materials/directions. */
    float alpha_max = 0.0f;
    for (int m = 0; m < NUM_MATERIALS; m++) {
        float ax = K_TABLE[m] / RHOCP_TABLE[m];
        float ay = KY_TABLE[m] / RHOCP_TABLE[m];
        if (ax > alpha_max) {
            alpha_max = ax;
        }
        if (ay > alpha_max) {
            alpha_max = ay;
        }
    }

    return 0.24f / (alpha_max * (1.0f / (dx * dx) + 1.0f / (dy * dy)));
}

static void assign_materials(Grid *g, int global_ny_start)
{
    /*
     * Layer map from bottom to top (global Y):
     *   0-40%   FR4 substrate
     *   40-50%  Copper spreader
     *   50-55%  TIM layer
     *   55-100% Silicon die
     */
    int fr4_end = (int)(0.40f * (float)g->ny);
    int copper_end = (int)(0.50f * (float)g->ny);
    int tim_end = (int)(0.55f * (float)g->ny);

    for (int j = 0; j <= g->local_ny + 1; j++) {
        int gy = global_ny_start + (j - 1);
        if (gy < 0) {
            gy = 0;
        }
        if (gy >= g->ny) {
            gy = g->ny - 1;
        }

        if (gy < fr4_end) {
            g->material_id[j] = MAT_FR4;
        } else if (gy < copper_end) {
            g->material_id[j] = MAT_COPPER;
        } else if (gy < tim_end) {
            g->material_id[j] = MAT_TIM;
        } else {
            g->material_id[j] = MAT_SILICON;
        }
    }
}

static void set_initial_condition(Grid *g, int global_ny_start)
{
    (void)global_ny_start;
    /* Start the full local slab (including ghost rows) at ambient temperature. */
    int nxy = g->nx * (g->local_ny + 2);
    for (int idx = 0; idx < nxy; idx++) {
        g->T_old[idx] = T_AMBIENT;
        g->T_new[idx] = T_AMBIENT;
    }
}

// === Section 2: Grid lifecycle ===

static void grid_init(Grid *g, int nx, int ny, int rank, int nranks)
{
    /* 1D slab decomposition along Y (each rank gets a contiguous row range). */
    int base = ny / nranks;
    int rem = ny % nranks;
    int global_ny_start;
    int nxy;

    memset(g, 0, sizeof(*g));
    g->nx = nx;
    g->ny = ny;
    g->rank = rank;
    g->nranks = nranks;

    if (rank < rem) {
        g->local_ny = base + 1;
        global_ny_start = rank * (base + 1);
    } else {
        g->local_ny = base;
        global_ny_start = rem * (base + 1) + (rank - rem) * base;
    }
    g->global_y_start = global_ny_start;

    /* Uniform spacing in both directions for this project phase. */
    g->dx = 1.0e-5f;
    g->dy = 1.0e-5f;
    g->dt = compute_dt(g->dx, g->dy);

    nxy = g->nx * (g->local_ny + 2);
    g->T_old = (float *)malloc((size_t)nxy * sizeof(float));
    g->T_new = (float *)malloc((size_t)nxy * sizeof(float));
    g->material_id = (int *)malloc((size_t)(g->local_ny + 2) * sizeof(int));

    if (g->T_old == NULL || g->T_new == NULL || g->material_id == NULL) {
        fprintf(stderr, "Rank %d: host allocation failed in grid_init.\n", rank);
        free(g->T_old);
        free(g->T_new);
        free(g->material_id);
        g->T_old = NULL;
        g->T_new = NULL;
        g->material_id = NULL;
        exit(EXIT_FAILURE);
    }

    /* Fill per-row material map and then initialize temperature field. */
    assign_materials(g, global_ny_start);
    set_initial_condition(g, global_ny_start);
}

static void grid_free(Grid *g)
{
    free(g->T_old);
    free(g->T_new);
    free(g->material_id);
    g->T_old = NULL;
    g->T_new = NULL;
    g->material_id = NULL;
}

static void grid_swap(Grid *g)
{
    /* Ping-pong host buffers after each step. */
    float *tmp = g->T_old;
    g->T_old = g->T_new;
    g->T_new = tmp;
#ifdef USE_CUDA
    /* Keep device pointers aligned with host ping-pong state. */
    tmp = g->d_T_old;
    g->d_T_old = g->d_T_new;
    g->d_T_new = tmp;
#endif
}

// === Section 3: I/O ===

static void make_outdir(const char *path)
{
    /* Ignore "already exists" so reruns append/overwrite snapshots naturally. */
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "mkdir failed for %s (errno=%d)\n", path, errno);
        exit(EXIT_FAILURE);
    }
}

static void write_metadata(Grid *g, int nsteps, int snap_every, const char *outdir)
{
    /* Single metadata file is written by rank 0 only. */
    if (g->rank != 0) {
        return;
    }

    char fname[512];
    snprintf(fname, sizeof(fname), "%s/metadata.txt", outdir);

    FILE *fp = fopen(fname, "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open %s for writing.\n", fname);
        return;
    }

    fprintf(fp, "nx=%d\n", g->nx);
    fprintf(fp, "ny=%d\n", g->ny);
    fprintf(fp, "nranks=%d\n", g->nranks);
    fprintf(fp, "dx=%.9e\n", g->dx);
    fprintf(fp, "dy=%.9e\n", g->dy);
    fprintf(fp, "dt=%.9e\n", g->dt);
    fprintf(fp, "nsteps=%d\n", nsteps);
    fprintf(fp, "snap_every=%d\n", snap_every);
    fclose(fp);
}

static void write_snapshot(Grid *g, int step, const char *outdir)
{
#ifdef USE_CUDA
    /* Snapshot writer always reads host memory, so mirror device -> host first. */
    cuda_copy_to_host(g);
#endif

    char fname[512];
    snprintf(fname, sizeof(fname), "%s/rank_%03d_step_%06d.bin", outdir, g->rank, step);

    FILE *fp = fopen(fname, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Rank %d: failed to open %s for writing.\n", g->rank, fname);
        return;
    }

    for (int j = 1; j <= g->local_ny; j++) {
        /* Skip ghost rows; write only owned physical rows. */
        size_t wrote = fwrite(&g->T_old[IDX(g, 0, j)], sizeof(float), (size_t)g->nx, fp);
        if (wrote != (size_t)g->nx) {
            fprintf(stderr, "Rank %d: short write at step %d.\n", g->rank, step);
            fclose(fp);
            return;
        }
    }

    fclose(fp);
}

// === Section 4: main() ===

int main(int argc, char **argv)
{
    Grid g;
    int nx;
    int ny;
    int nsteps;
    int snap_every;
    char *outdir;
    int rank = 0;
    int nranks = 1;
    double t0;
    double t1;
    double elapsed;
    double wall_time;

    parse_args(argc, argv, &nx, &ny, &nsteps, &snap_every, &outdir);

    /* Initialize distributed runtime only when MPI build is enabled. */
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif

#ifdef USE_CUDA
    {
        /* One GPU per rank (round-robin if ranks > devices). */
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            fprintf(stderr, "No CUDA devices available.\n");
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return EXIT_FAILURE;
        }
        CUDA_CHECK(cudaSetDevice(rank % device_count));
    }
#endif

    grid_init(&g, nx, ny, rank, nranks);

    /* Allocate optional accelerator / communication resources. */
#ifdef USE_CUDA
    cuda_alloc(&g);
#endif
#ifdef USE_MPI
    halo_alloc(&g);
#endif

    make_outdir(outdir);
    write_metadata(&g, nsteps, snap_every, outdir);
    write_snapshot(&g, 0, outdir);

    /* Start timing after all setup so runtime is pure simulation loop cost. */
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
#else
#ifdef USE_OMP
    t0 = omp_get_wtime();
#else
    t0 = (double)clock() / (double)CLOCKS_PER_SEC;
#endif
#endif

    for (int step = 1; step <= nsteps; step++) {
        /* Choose compute backend at compile time. */
#if defined(USE_CUDA)
        launch_stencil_cuda(&g);
#elif defined(USE_OMP)
        apply_stencil_omp(&g);
#else
        apply_stencil_serial(&g);
#endif

#ifdef USE_CUDA
        /* Keep this explicit so halo exchange always sees finished kernel writes. */
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_MPI
        /* Exchange top/bottom boundary rows between neighboring ranks. */
#ifdef USE_CUDA
        halo_exchange_hybrid(&g);
#else
        halo_exchange_cpu(&g);
#endif
#endif

        grid_swap(&g);

        /* Periodic snapshots for visualization and verification. */
        if (step % snap_every == 0) {
            write_snapshot(&g, step, outdir);
        }
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
#else
#ifdef USE_OMP
    t1 = omp_get_wtime();
#else
    t1 = (double)clock() / (double)CLOCKS_PER_SEC;
#endif
#endif

    elapsed = t1 - t0;

#ifdef USE_MPI
    /* Wall time is the max rank time, not rank 0 local time. */
    MPI_Reduce(&elapsed, &wall_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
    wall_time = elapsed;
#endif

    if (rank == 0) {
        const char *version;
#if defined(USE_MPI) && defined(USE_CUDA)
        version = "hybrid";
#elif defined(USE_CUDA)
        version = "cuda";
#elif defined(USE_OMP)
        version = "omp";
#else
        version = "serial";
#endif
        printf("%s,%d,%d,%d,%d,%.6f\n", version, nx, ny, nranks, nsteps, wall_time);
    }

    /* Cleanup in strict reverse order of allocation/init. */
#ifdef USE_MPI
    halo_free(&g);
#endif
#ifdef USE_CUDA
    cuda_free(&g);
#endif
    grid_free(&g);

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
