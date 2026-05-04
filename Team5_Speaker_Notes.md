# Speaker Talking Lines — Team 5
## *Accelerated Thermal Simulation of High Power Electronic Systems*
COMPE596 — Spring 2026

---

## A note on 2D vs. 3D before you start

You will get questions on this. The **original project proposal in our notes was 3D** — a full multi-layer package with stacking in the through-package direction (Z), Y-slab decomposition across MPI ranks, and a 3D stencil with six neighbors per cell. We scoped down to **2D cross-section** for four concrete, defensible reasons:

1. **Memory footprint.** A 3D stencil at 4096³ in `float` is 4096 × 4096 × 4096 × 4 bytes ≈ **256 GB per buffer**, and we need two buffers (`T_old`, `T_new`). That does not fit on a single A100 (80 GB) and barely fits across the DGX node. A 2D 4096² grid is 64 MB per buffer — comfortable. Even our largest run, 65536², is only ~16 GB per buffer, which fits on one GPU.
2. **Stencil arithmetic intensity.** The 5-point 2D stencil has ~5 loads + 1 store per cell. The 7-point 3D stencil has ~7 loads + 1 store. The 3D version is more memory-bandwidth-bound on the GPU and produces *less impressive* speedup curves for the same engineering effort. A clean 2D run shows the parallelization story better.
3. **The physics still captures hot-spot formation.** A 2D vertical cross-section through the package preserves what we actually care about: heat moving from a silicon die down through TIM, copper, and FR4. The anisotropic conductivity tensor (different `k_x` and `k_y`) is still meaningful because FR4 and copper traces are physically anisotropic in the cross-section plane. We lose lateral spreading in the third direction, but for showing thermal coupling between layers, 2D is sufficient.
4. **Time budget.** Tier A in our scope was 2D. Tier B (3D) was a stretch goal. We executed Tier A cleanly across all three parallel technologies (CUDA + MPI + OpenMP) instead of half-implementing 3D.

**One-sentence answer if asked:** *"We chose 2D so the entire problem fits comfortably in GPU memory at our largest grid sizes, which let us actually push the scaling study out to 65k² instead of being capped by VRAM at much smaller sizes — and the physics of layered hot-spot formation is still captured in a vertical cross-section."*

---

## Slide 1 — Title

> "Good morning. We're Undergraduate Team 5 — Ulises, Brandon, Jimin, Christopher, and Yousif — and our project is *Accelerated Thermal Simulation of High Power Electronic Systems*. The short version is that we took the heat equation that governs how a CPU or GPU die heats up under load, and we accelerated it three different ways at once: CUDA on the GPU, MPI across multiple GPUs, and OpenMP on the CPU. The goal was to take a thermal simulation that runs in over two and a half hours on a single CPU core and get it down to something practical, while preserving correctness against a serial reference."

---

## Slide 2 — Physical Problem

> "The physical motivation is real and current. Modern processors dissipate hundreds of watts in dies that are only a few millimeters across. Wherever transistor density spikes — the cores, the FPU, the L2 controllers — you get localized hot spots, and if those aren't managed, the chip throttles, electromigration accelerates, and eventually the device fails. So we try to simulate where heat actually goes inside the package."

> "Within a 2D vertical cross-section of a real IC stack-up. Heat starts at the silicon die at the top, then has to travel through a thin layer of thermal interface material — the TIM — into a copper heat-spreader, and finally down into the FR4 substrate underneath. Each of those four layers has *anisotropic* thermal conductivity, meaning heat doesn't flow at the same rate in the in-plane and through-plane directions — that's important for FR4 because the glass-fiber weave conducts heat differently along the board than across it. So our stencil update has to look up material properties cell-by-cell, which makes this more computationally intensive."

*(If asked why 2D instead of 3D: we chose 2D because it fits comfortably in GPU memory at our largest grid sizes so we could run a full scaling study up to 65,536², while still capturing the key layered hot-spot physics in a vertical cross-section.)*

---

## Slide 3 — Domain Parameters

> "Quick tour of the simulation domain. Our default grid is 4096 by 4096 cells, but we built the run system to scale all the way up to 65,536 by 65,536 — that's the largest size in our scaling study, and it's where the parallel speedup numbers really start to mean something. Cell spacing is 10 microns in both directions, which gives us realistic die-scale resolution."
>
> "We run up to ten thousand explicit time steps, with all temperatures initialized to ambient at 300 Kelvin. The vertical layer stack goes from FR4 substrate at the bottom — 0 to 40 percent of the domain — copper heat-spreader from 40 to 50 percent, TIM from 50 to 55, and the silicon die occupying the top 45 percent. We seed three Gaussian heat sources at 25, 50, and 75 percent of the die width — those are our hot spots, modeling a real multi-core processor floor plan with three high-power regions. Everything you see here is parameterized in either the Makefile or in `grid_init` and `assign_materials`, so the same code runs on all our scaling sizes without source changes."

---

## Slide 4 — Mathematical Models

> "This is the heat equation we solve. Left side first: `ρ c_p * dT/dt`. Here `ρ` is density, `c_p` is specific heat, and `ρ c_p` together is volumetric heat capacity — basically thermal inertia. So that term says how hard it is to change temperature in each material."
>
> "Right side is heat spreading and heat generation. `k_x * d²T/dx²` is conduction in x, `k_y * d²T/dy²` is conduction in y, and `Q(x,y)` is the heat source from our hotspot power. Having separate `k_x` and `k_y` is anisotropy. For FR4, `k_x = 0.8` and `k_y = 0.3`, so heat spreads much better in one direction than the other."
>
> "In plain terms, each grid cell updates from itself and its four neighbors: up, down, left, and right. We advance in small time steps so the simulation stays stable, with the step size chosen based on the fastest-spreading material, which is copper. Since copper moves heat far faster than FR4, there is no simple one-line solution, so we solve it numerically."

---

## Slide 5 — Code Implementation

> "A few implementation choices that matter for performance. First, even though the grid is logically 2D, we store it as a *flattened 1D array* — that's the `IDX(g, i, j)` macro you'll see throughout the code. That gives the compiler — and the GPU — clean, contiguous memory access patterns that are friendly to index."
>
> "Second, we use double-buffering: `T_old` for reads and `T_new` for writes, swapped every step. This is essential, With two buffers, the math is safe to parallelize across all cells simultaneously, which is what makes both OpenMP and CUDA possible."
>
> "Third, we determine each row's material once at startup and then reuse that lookup on every time step. That keeps the inner loop simple and fast because we avoid re-deciding material thousands of times. We model power as three Gaussian hotspots at the core locations, and we apply that source only in silicon cells where active devices live. Since many threads in the same row follow the same material rule, the GPU executes this branch efficiently. Finally, we write timing outputs in CSV format so plotting scripts can consume results automatically."

---

## Slide 6 — Serial Implementation

> "This is our baseline — pure serial C, one CPU core, no parallelism. It exists for two reasons: first, it's the *correctness oracle*. Every other version's output is diff'ed against this one to make sure parallelism didn't break the physics. Second, it's the denominator in every speedup number on this presentation."
>
> "The structure is straightforward: outer loop over rows from j equals 1 to local_ny, inner loop over columns from 1 to nx-2. We skip index 0 and the last index because those are Dirichlet boundaries — fixed at ambient — and skipping them in serial means the parallel versions don't have to special-case them either. That's what we mean by 'boundary handling consistent with parallel versions for fair comparison.'"
>
> "Inside `update_cell`, you can see the actual physics: we look up the material ID for this row, fetch k_x, k_y, and ρ·cₚ from the material tables, compute the diffusivities ax and ay, evaluate the hot-spot source, load the five neighbors, and compute the new value with the explicit forward-Euler step. This is the function that the OpenMP and CUDA versions are racing against."

---

## Slide 7 — OpenMP Implementation

> "OpenMP was the smallest code change for the largest first improvement. The math is identical to serial — we use the same `update_cell` function, untouched. The only difference is one line: `#pragma omp parallel for schedule(static)` on the outer loop over rows."
>
> "What that pragma does mechanically: at runtime the OpenMP runtime spawns a thread team — controlled by the `OMP_NUM_THREADS` environment variable — and divides the row range evenly across them with `schedule(static)`. Each thread owns a contiguous block of rows and updates them independently. Because we use double-buffering, no two threads ever write to the same cell, so we don't need locks or atomics."
>
> "The `#ifdef USE_OMP` guard is intentional. If the executable is built without OpenMP enabled, we fall through to the serial path — same source file, three different binaries depending on Makefile targets. We see good acceleration up to about 16 threads on the DGX nodes, but past that we hit the memory bandwidth wall — the stencil is bandwidth-bound on the CPU because every cell requires loading five floats from DRAM. That's the natural ceiling that motivates moving to the GPU."

---

## Slide 8 — CUDA Implementation

> "CUDA is where we actually break through the bandwidth ceiling, because GPU memory bandwidth is roughly an order of magnitude higher than CPU. The launch model is one CUDA thread per interior cell. Block dimensions default to 16 by 16 — that's our `TILE_WIDTH` — but they're tunable from the command line via `--block-x` and `--block-y`."
>
> "We ship two kernels in the same source file. The first, `stencil_kernel_naive`, is the literal port of the serial code — every thread reads its five neighbors directly from global memory. That's what you see on the right of the slide. It already gets us about a 4 to 5x speedup over serial because of GPU bandwidth alone."
>
> "The second kernel, `stencil_kernel_tiled`, uses shared memory. Each thread block cooperatively loads a `(TILE_WIDTH+2)` by `(TILE_WIDTH+2)` patch into shared memory — that's the inner tile plus a one-cell halo on every side — and then every thread reads its five neighbors from shared memory instead of global. That eliminates redundant global loads — without tiling, each interior cell gets read by five different threads as a neighbor. With tiling, it's read once into shared memory and reused. We use `__syncthreads()` to make sure the tile is fully loaded before any thread reads from it."
>
> "One important detail: in pure CUDA mode, the temperature buffers and the material/property tables all live on the device. We don't shuttle data between host and GPU every time step. Host traffic only happens for snapshots and at the very end — that's why CUDA scales so well on large grids."

---

## Slide 9 — Hybrid Implementation (CUDA + MPI + OpenMP)

> "This is the rubric piece — all three technologies running together in one program. The decomposition is *Y-slab*: we split the global grid horizontally into strips, and each MPI rank owns one strip. With two ranks, the top half goes to rank 0, the bottom half to rank 1. With four ranks, we split into quarters."
>
> "Each rank has *two extra rows* on top of the rows it owns — those are the ghost rows, also called halo rows. They're copies of the neighboring rank's edge data, and they have to be refreshed every time step because the stencil reaches across that boundary."
>
> "The exchange protocol, implemented in `halo.c`, is: after every CUDA stencil step, we `cudaDeviceSynchronize` to make sure the kernel has finished writing. Then we copy `d_T_old` from device to host, do an `MPI_Sendrecv` of the boundary rows between adjacent ranks — top rank sends its bottom row down and receives a new top row from below, and vice versa — and finally we copy the updated host buffer back to the device. So every step has a device-to-host copy, an MPI exchange, and a host-to-device copy. That's communication overhead, and it's why the hybrid speedup is bigger than CUDA-alone only at large grid sizes — at small sizes, the halo exchange dominates."
>
> "OpenMP shows up on the host side: we use it to parallelize the halo packing on the CPU before the MPI call, which is small but free if you're already linked against OpenMP."
>
> "And `extern \"C\"` matters here — `stencil_cuda.cu` is compiled by `nvcc` as C++, but `main.c` and `halo.c` are compiled by `mpicc` as C. The `extern \"C\"` linkage on the launcher prevents C++ name-mangling so the C linker can find the symbol."
>
> "The numbers on the slide are an early result: 1024², 100 steps, 2 ranks. Serial took 2.4 seconds, hybrid took 0.09 seconds — about 26x speedup. We'll show the full scaling study on the next slides."

---

## Slide 10 — Results

> "This is our scaling study, six grid sizes from 1024² up to 65,536². I'll walk through the trend rather than every number. Three things to notice."
>
> "First, *all three accelerated versions beat serial at every grid size* — the speedups are real, not artifacts. At 4096², serial takes 41 seconds, OpenMP takes 14, CUDA takes 7.5, and hybrid takes 5.8. The hybrid is fastest because it has both more compute throughput — two GPUs — and aggregated VRAM."
>
> "Second, *the speedup ratio grows with grid size*, which is exactly what you'd predict from Amdahl's law plus the GPU's memory hierarchy. At 1024², hybrid is only 5.3x faster than serial because the kernel launch and halo exchange overhead is a big fraction of the runtime. At 16384², hybrid is 12.5x faster because the actual compute time has grown to dominate that fixed overhead."
>
> "Third, look at 32768²: serial takes over 43 *minutes*. The hybrid version finishes in under 4 minutes. And at 65,536² — that's a four-billion-cell grid — serial takes nearly *2 hours and 47 minutes*; the OpenMP run with 32 threads finishes in 11 minutes, a 14.7x speedup. That's the regime where this kind of acceleration becomes the difference between 'we can iterate on a thermal design today' and 'we run it overnight and hope.'"
>
> "One honest caveat on 65,536²: the hybrid number actually *underperforms* OpenMP-32 at that size. We think this is because the halo-exchange traffic — D2H, MPI send/recv, H2D — at that grid size is exchanging 256 KB rows every step, and that's saturating the PCIe link. It's a known limitation of the simple unpinned-buffer halo path, and it's the obvious next thing to optimize."

---

## Slide 11 — Full Results Table

> "This is the raw CSV behind the previous slide — every variant, every grid size, every thread count and block size we benchmarked. I won't read it; the columns are: variant, nx, ny, MPI ranks, time steps, OMP threads, block size, samples taken, wall time in seconds, and speedup against serial at the same grid size."
>
> "Two things worth pointing out. The OpenMP rows show clean thread scaling — at 8192², going from 4 threads to 32 threads moves wall time from 104 seconds to 39 seconds, a 2.7x improvement on top of the parallelization itself. That tells us OpenMP is genuinely using the cores, not getting lock-contended."
>
> "And the hybrid rows show that *more ranks isn't always better*. At 4 ranks with 32 threads at 32768², we get 11.6x speedup. At 16 threads we get 11.8x. At 8 threads, 11.7x. They're essentially flat because beyond a point, the halo exchange and not the compute dominates — adding more host threads can't help that. So this table is also a guide for future work on overlapping communication with computation."

---

## Slide 12 — Scaling Plots

> "Two plots. The top one is wall time on a log-log scale. The serial line is the dashed black reference, and you can see all the parallel versions sitting below it at every grid size. CUDA — the red line — is a clean straight line, which means it's scaling linearly with the number of cells. The hybrid lines, in green, drop *below* the CUDA line at large grid sizes — that's the second GPU paying off."
>
> "The bottom plot is the same data expressed as relative speedup over serial. The interesting story is that *no single configuration is best at every grid size*. CUDA and hybrid dominate from 1024² to 32768². But at 65536², OpenMP with 32 threads jumps to a 14.7x speedup and pulls ahead of hybrid. That's the PCIe-saturation effect I mentioned — at the largest grid, the halo bytes per step are big enough that staying on the CPU and skipping the host-device copies entirely turns out to be faster than the hybrid path."
>
> "The takeaway is the standard one for HPC: the right tool depends on the problem size. For mid-sized grids, GPU + MPI hybrid wins. For very small grids, OpenMP alone wins because launch overhead dominates. For grids large enough to stress PCIe, a fat CPU node with high thread counts is competitive again — and on a real industrial cluster you'd want pinned host memory and CUDA-aware MPI, both of which are clear extensions of this work."

---

## Slide 13 — Thank You / Q&A

> "Thanks. Happy to take questions."

---

## Anticipated Q&A — quick answers

**"Why explicit time integration instead of implicit?"**
> Explicit is dramatically simpler — every cell update is independent, which is what makes the stencil parallelize cleanly across MPI, OpenMP, and CUDA. Implicit (Crank-Nicolson, ADI) would require solving a sparse linear system every step, which would mean pulling in a parallel linear solver and would defeat the rubric's "implement parallelism yourself" intent. The cost is a CFL-restricted Δt, but for our diffusivities and 10 µm cell spacing the stable Δt is small enough that 10,000 steps still resolves real thermal transients on the order of milliseconds.

**"How do you know the answer is correct?"**
> Two ways. First, every parallel variant's binary output is diff'ed against the serial baseline using `verify/diff_outputs.py` — they match within float rounding. Second, the steady-state temperature distribution makes physical sense: the silicon die is hottest, the copper spreader stays nearly isothermal because of its conductivity, and FR4 acts as a thermal insulator below.

**"Why `float` and not `double`?"**
> Float is faster on the GPU and uses half the memory bandwidth. We checked that the answer doesn't drift catastrophically over 10,000 steps by comparing to a `double` CPU reference run — the difference is in the fifth decimal place, well below the precision we'd care about for a thermal design.

**"Why two GPUs and not eight?"**
> The DGX has eight GPUs, but our halo exchange is over `MPI_Sendrecv` through host memory — it's not CUDA-aware MPI and it's not using NVLink directly. Past 4 GPUs the host-side copy is the bottleneck and adding more ranks doesn't help. Real CUDA-aware MPI with GPUDirect would let us scale further, and that's listed as future work.

**"What was the hardest debugging issue?"**
> Getting `extern "C"` linkage right between the `nvcc`-compiled `.cu` file and the `mpicc`-compiled `.c` files. The first build failed at link time with unresolved symbols because C++ was mangling the launcher names. Wrapping the function declarations in `#ifdef __cplusplus` / `extern "C"` blocks in `heat.h` fixed it.

**"What would you do with another two weeks?"**
> Three things, in priority order. One, switch the halo exchange to pinned host memory and CUDA-aware MPI to remove the PCIe bottleneck at 65k². Two, overlap halo communication with interior-cell computation using CUDA streams — there's no reason the GPU should be idle during the MPI call. Three, extend to 3D so we can capture lateral spreading in the third direction, which is the missing physics in the current cross-section model.
