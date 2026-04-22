#include "heat.h"

/*
 * This file contains the CPU-side stencil update for our package-thermal model.
 * PDE form (2D anisotropic diffusion):
 *     dT/dt = ax * d2T/dx2 + ay * d2T/dy2
 * with explicit Euler time integration:
 *     T_new = T_old + dt * (ax * Laplacian_x + ay * Laplacian_y)
 */

/* In-plane conductivity (W/mK): FR4, Copper, TIM, Silicon. */
const float K_TABLE[NUM_MATERIALS] = {0.8f, 400.0f, 5.0f, 148.0f};
/* Through-plane conductivity (W/mK): FR4 is anisotropic (lower in Y here). */
const float KY_TABLE[NUM_MATERIALS] = {0.3f, 400.0f, 5.0f, 148.0f};
/* Volumetric heat capacity (J/m^3K): rho * cp for each material. */
const float RHOCP_TABLE[NUM_MATERIALS] = {2.4e6f, 3.45e6f, 1.75e6f, 1.63e6f};

/* Continuous volumetric heat source (W/m^3) for silicon hot-spot clusters. */
static float hotspot_power_density(const Grid *g, int i, int gy, int material)
{
    if (material != MAT_SILICON) {
        return 0.0f;
    }

    float sigma = (float)g->nx * HOTSPOT_SIGMA_FRAC;
    float two_sigma2 = 2.0f * sigma * sigma;
    float cx0 = 0.25f * (float)g->nx;
    float cx1 = 0.50f * (float)g->nx;
    float cx2 = 0.75f * (float)g->nx;
    float cy = HOTSPOT_Y_FRAC * (float)g->ny;
    float dy0 = (float)gy - cy;
    float dx0 = (float)i - cx0;
    float dx1 = (float)i - cx1;
    float dx2 = (float)i - cx2;

    return HOTSPOT_POWER_PEAK *
           (expf(-(dx0 * dx0 + dy0 * dy0) / two_sigma2) +
            expf(-(dx1 * dx1 + dy0 * dy0) / two_sigma2) +
            expf(-(dx2 * dx2 + dy0 * dy0) / two_sigma2));
}

/*
 * Single-cell update helper so serial/OMP paths stay easy to read.
 * We keep all neighbor accesses explicit for clarity and debugging.
 */
static void update_cell(Grid *g, int i, int j)
{
    int m = g->material_id[j];
    int gy = g->global_y_start + (j - 1);
    float kx = K_TABLE[m];
    float ky = KY_TABLE[m];
    float rhocp = RHOCP_TABLE[m];
    float ax = kx / rhocp;
    float ay = ky / rhocp;
    float qdot = hotspot_power_density(g, i, gy, m);

    float c = g->T_old[IDX(g, i, j)];
    float n = g->T_old[IDX(g, i, j + 1)];
    float s = g->T_old[IDX(g, i, j - 1)];
    float e = g->T_old[IDX(g, i + 1, j)];
    float w = g->T_old[IDX(g, i - 1, j)];

    g->T_new[IDX(g, i, j)] =
        c + g->dt * (ax * (e - 2.0f * c + w) / (g->dx * g->dx) +
                     ay * (n - 2.0f * c + s) / (g->dy * g->dy) +
                     qdot / rhocp);
}

/* Serial baseline: updates interior points, leaves left/right boundaries untouched. */
void apply_stencil_serial(Grid *g)
{
    /* j=1..local_ny skips ghost rows at 0 and local_ny+1. */
    for (int j = 1; j <= g->local_ny; j++) {
        /* i=1..nx-2 preserves Dirichlet left/right edges. */
        for (int i = 1; i < g->nx - 1; i++) {
            update_cell(g, i, j);
        }
    }
}

/* OpenMP version: same math, parallelized over slab rows. */
void apply_stencil_omp(Grid *g)
{
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
    for (int j = 1; j <= g->local_ny; j++) {
        for (int i = 1; i < g->nx - 1; i++) {
            update_cell(g, i, j);
        }
    }
#else
    /* If OpenMP is not enabled, keep behavior valid by using serial path. */
    apply_stencil_serial(g);
#endif
}
