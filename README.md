# Accelerated Thermal Simulation of High-Power Electronic Systems

This project models how heat spreads through a layered electronic package (chip + package materials) using a 2D anisotropic heat equation.  
In simple words: we place temperature values on a large grid, then repeatedly update each cell using its nearby neighbors to simulate physical heat flow over time.

The same physics is implemented in four execution modes so we can compare correctness and speed:

- serial CPU
- OpenMP multi-threaded CPU
- CUDA GPU
- hybrid MPI + CUDA + OpenMP (1D Y-slab decomposition, one rank per GPU)

---

## 1) What problem this project solves

Modern high-power chips can generate large thermal gradients. If some regions run hotter than others, performance and reliability drop.  
Before hardware is built, thermal simulation helps engineers answer questions like:

- Where will the hottest spot be?
- How quickly will heat spread?
- How does material anisotropy change temperature distribution?
- How much runtime can we reduce with parallel hardware?

This repository focuses on both:

1. **Physics fidelity at stencil level** (anisotropic diffusion with layered properties).
2. **High-performance implementation** across CPU, GPU, and distributed GPU systems.

---

## 2) Physical model (easy explanation + theory)

### 2.1 Continuous equation

The temperature field is `T(x, y, t)`.  
For anisotropic diffusion in 2D, the governing form is:

```text
∂T/∂t = ∂/∂x (alpha_x * ∂T/∂x) + ∂/∂y (alpha_y * ∂T/∂y) + Q
```

Where:

- `alpha_x`, `alpha_y` are thermal diffusivities in x and y directions.
- Anisotropic means heat can spread at different rates in different directions.
- `Q` is an optional heat source term (depends on configuration).

In layered package materials, coefficients vary by region, so some zones conduct faster/slower.

### 2.2 Why matrices appear in code

The continuous PDE cannot be solved directly on infinite points, so we discretize space into a rectangular grid:

- `nx` points in x
- `ny` points in y

This produces a temperature **matrix** `T[j][i]` (or flattened 1D array with 2D indexing).  
Each time step computes a new matrix from the old matrix:

```text
T_next = F(T_curr)
```

This is the key computational pattern in all variants.

### 2.3 Finite-difference stencil idea

For each interior cell `(i, j)`, the update uses the center plus neighbors:

- left/right neighbors for x diffusion
- up/down neighbors for y diffusion

This local pattern is called a **stencil**.  
Because each new cell value depends only on nearby old values, many cells can be updated in parallel.

### 2.4 Time stepping intuition

The simulation loop is:

1. Read current temperature matrix.
2. Compute next temperature matrix with stencil updates.
3. Swap pointers (next becomes current).
4. Repeat for `steps`.

So the code performs iterative relaxation of temperature over time, converging toward physically expected diffusion behavior.

---

## 3) How parallelization works in this project

The same stencil math is parallelized at different scales.

### 3.1 Serial CPU

- One core updates every grid point in nested loops.
- Baseline for correctness and speedup comparisons.

### 3.2 OpenMP CPU (shared memory)

- The same loops are split across CPU threads.
- Different threads update different rows/blocks concurrently.
- Best for multi-core nodes when GPU is not used.

### 3.3 CUDA GPU (massive thread parallelism)

- Grid cells are mapped to many GPU threads.
- Each thread computes one (or a few) output cells.
- Tiled/shared-memory kernels reduce global memory traffic by reusing neighbor data.

Why this is effective:

- Stencil updates are arithmetic-light but memory-heavy.
- GPU memory bandwidth + many threads improve throughput for large grids.

### 3.4 Hybrid MPI + CUDA + OpenMP (distributed domain decomposition)

For very large grids and multi-GPU scaling:

- Domain is split across MPI ranks in **1D Y-slabs**.
- Each rank owns a contiguous chunk of rows.
- One rank controls one GPU.

Boundary handling across ranks:

- Each rank needs one neighbor row from rank above and below.
- Halo (ghost) rows are exchanged every step using MPI.
- After halo exchange, each rank runs local CUDA stencil on its slab.

This gives two-level parallelism:

1. **Inter-node/inter-rank parallelism** via MPI decomposition.
2. **Intra-rank device parallelism** via CUDA kernels (plus optional CPU threading around host work).

---

## 4) Theoretical reasoning for performance behavior

### 4.1 Computational complexity

Per step, work is proportional to number of cells:

```text
O(nx * ny)
```

Total runtime is approximately:

```text
O(nx * ny * steps)
```

So doubling both `nx` and `ny` roughly quadruples cell updates per step.

### 4.2 Memory-bound nature

Stencil methods usually load several neighbor values and write one result, so memory movement dominates arithmetic.  
Optimizations therefore focus on:

- contiguous access
- cache reuse (CPU)
- shared-memory tiling/coalescing (GPU)
- minimizing communication overhead (MPI halos)

### 4.3 Communication vs computation in MPI

Each rank computes on its local slab but communicates boundaries each step.  
As rank count increases:

- compute per rank decreases
- communication per rank does not shrink as quickly

This creates a scaling tradeoff: strong scaling eventually becomes communication-limited.

### 4.4 Why 1D Y-slab decomposition

A 1D slab split is simple and efficient for this stencil:

- only top/bottom neighbor exchange
- low implementation complexity
- good fit for row-major memory layouts

It is a practical first decomposition for multi-GPU heat diffusion.

---

## 5) Source code map

Main source files in `src/`:

- `heat.h`  
  Shared constants, `Grid` data structure, parameter definitions, and function declarations.

- `main.c`  
  Program entry and orchestration: parse args, initialize fields, run timestep loop, time execution, write snapshots and CSV timing row.

- `stencil_cpu.c`  
  CPU stencil implementations (`apply_stencil_serial`, `apply_stencil_omp`) and material-table logic.

- `stencil_cuda.cu`  
  CUDA kernels (naive and tiled variants), launch wrappers, and device memory helpers.

- `halo.c`  
  MPI halo exchange routines for rank-to-rank top/bottom ghost rows.

- `Makefile`  
  Build rules for all execution modes and scaling-study workflow.

- `visualize.py`  
  Timing summary and heatmap visualization modes.

---

## 6) Build and run

From your Linux environment (for example through MobaXterm):

```bash
cd ~/Accelerated-Thermal-Simulation-of-High-Power-Electronic-Systems/src
```

### 6.1 Build all targets

```bash
make clean
make
```

Produces:

- `heat_serial`
- `heat_omp`
- `heat_cuda`
- `heat_hybrid`

### 6.2 Run examples

Serial CPU:

```bash
./heat_serial --nx 1024 --ny 1024 --steps 1000 --snap-every 200 --outdir results_serial
```

OpenMP CPU:

```bash
export OMP_NUM_THREADS=16
./heat_omp --nx 1024 --ny 1024 --steps 1000 --snap-every 200 --outdir results_omp
```

CUDA GPU:

```bash
./heat_cuda --nx 4096 --ny 4096 --steps 1000 --snap-every 250 --outdir results_cuda
```

Hybrid MPI + CUDA + OpenMP:

```bash
export OMP_NUM_THREADS=8
mpirun -np 2 ./heat_hybrid --nx 4096 --ny 4096 --steps 1000 --snap-every 250 --outdir results_hybrid
```

---

## 7) GPU architecture selection

Default build architecture is `ARCH=sm_80` (A100 class GPU).  
For V100 (`sm_70`):

```bash
make clean
make ARCH=sm_70
```

---

## 8) Scaling workflow and visualization

Run an automated sweep and summary:

```bash
make scaling-study
```

This appends timing rows to `results/scaling.csv` and executes:

```bash
python3 visualize.py --mode scaling
```

Manual commands:

```bash
python3 visualize.py --mode scaling --csv results/scaling.csv
python3 visualize.py --mode heatmap --outdir results_cuda
python3 visualize.py --mode heatmap --outdir results_hybrid
python3 visualize.py --mode heatmap --outdir results_cuda --cmap inferno
```

---

## 9) Output format and files

### 9.1 Terminal CSV line (one row per run)

```text
version,nx,ny,nranks,nsteps,wall_time_s
```

Example:

```text
cuda,4096,4096,1,1000,1.234567
```

`version` is one of: `serial`, `omp`, `cuda`, `hybrid`.

### 9.2 Files written to `--outdir`

- `metadata.txt`  
  Stores configuration and derived values (`nx`, `ny`, `nranks`, `dx`, `dy`, `dt`, `nsteps`, `snap_every`).

- `rank_RRR_step_SSSSSS.bin`  
  Binary float temperature snapshots, one file per rank and snapshot step.

Snapshots are written at step `0` and every `snap_every` steps.

---

## 10) Parameter guidance

- `--nx`, `--ny`: spatial resolution (matrix size).  
  Bigger values improve detail but increase runtime and memory use.

- `--steps`: number of time iterations.  
  More steps simulate longer time and cost more compute.

- `--snap-every`: output frequency.  
  Smaller value means more files and higher I/O overhead.

Suggested presets:

- quick debug: `--nx 512 --ny 512 --steps 200 --snap-every 100`
- CPU comparison: `--nx 1024 --ny 1024 --steps 1000 --snap-every 200`
- GPU/hybrid benchmark: `--nx 4096 --ny 4096 --steps 1000 --snap-every 250`

---

## 11) Practical interpretation of results

When analyzing output:

- Compare `wall_time_s` across versions for same `nx`, `ny`, `steps`.
- Verify temperature fields remain physically smooth (no unstable oscillations).
- Check scaling efficiency: GPU should improve single-node throughput; MPI+GPU should improve multi-GPU throughput until communication overhead dominates.

In short, this codebase is both a thermal simulation tool and a performance study platform for matrix-based stencil solvers on modern heterogeneous hardware.
