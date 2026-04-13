#include "heat.h"

/*
 * MPI halo exchange for Y-slab decomposition.
 *
 * Layout reminder:
 *   - local physical rows are j = 1 .. local_ny
 *   - top ghost row is j = 0
 *   - bottom ghost row is j = local_ny + 1
 *
 * We exchange rows of T_old after each stencil step so neighbors have fresh
 * boundary values for the next time step.
 */

#ifdef USE_MPI

void halo_alloc(Grid *g)
{
    /* No extra buffers needed: each row is contiguous in memory already. */
    (void)g;
}

void halo_free(Grid *g)
{
    /* No allocated halo state in this simple implementation. */
    (void)g;
}

void halo_exchange_cpu(Grid *g)
{
    if (g->nranks <= 1) {
        return;
    }

    int up = (g->rank == 0) ? MPI_PROC_NULL : (g->rank - 1);
    int down = (g->rank == g->nranks - 1) ? MPI_PROC_NULL : (g->rank + 1);
    MPI_Status status;

    /*
     * Exchange 1:
     *   send first owned row upward
     *   receive bottom ghost row from downward neighbor
     */
    MPI_Sendrecv(&g->T_old[IDX(g, 0, 1)], g->nx, MPI_FLOAT, up, 0,
                 &g->T_old[IDX(g, 0, g->local_ny + 1)], g->nx, MPI_FLOAT, down, 0,
                 MPI_COMM_WORLD, &status);

    /*
     * Exchange 2:
     *   send last owned row downward
     *   receive top ghost row from upward neighbor
     */
    MPI_Sendrecv(&g->T_old[IDX(g, 0, g->local_ny)], g->nx, MPI_FLOAT, down, 1,
                 &g->T_old[IDX(g, 0, 0)], g->nx, MPI_FLOAT, up, 1,
                 MPI_COMM_WORLD, &status);
}

void halo_exchange_hybrid(Grid *g)
{
#ifdef USE_CUDA
    /*
     * Simple and readable hybrid path:
     *   1) mirror d_T_old -> T_old
     *   2) perform CPU MPI halo exchange on host
     *   3) mirror T_old -> d_T_old
     *
     * This is not the fastest path, but it is easy to verify and debug.
     */
    size_t bytes = (size_t)g->nx * (size_t)(g->local_ny + 2) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(g->T_old, g->d_T_old, bytes, cudaMemcpyDeviceToHost));
    halo_exchange_cpu(g);
    CUDA_CHECK(cudaMemcpy(g->d_T_old, g->T_old, bytes, cudaMemcpyHostToDevice));
#else
    halo_exchange_cpu(g);
#endif
}

#else

/*
 * Non-MPI fallback stubs: keep linkage valid if this file is compiled
 * in environments without MPI enabled.
 */
void halo_alloc(Grid *g) { (void)g; }
void halo_free(Grid *g) { (void)g; }
void halo_exchange_cpu(Grid *g) { (void)g; }
void halo_exchange_hybrid(Grid *g) { (void)g; }

#endif
