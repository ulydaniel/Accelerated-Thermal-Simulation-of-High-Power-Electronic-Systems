# ============================================================================
# Hybrid heat diffusion build file
# ----------------------------------------------------------------------------
# Why compile in two phases?
#   1) Compile each source into an object file (.o)
#   2) Link object files into an executable
#
# Which compiler handles each file type?
#   - main.c / stencil_cpu.c / halo.c  : C compilers (gcc or mpicc)
#   - stencil_cuda.cu                   : nvcc
#
# Which linker builds each executable?
#   - heat_serial, heat_omp             : gcc
#   - heat_cuda                         : nvcc
#   - heat_hybrid (MPI + CUDA + OMP)    : mpicxx with -lcudart
# ============================================================================

SHELL ?= /bin/bash

CC ?= gcc
MPICC ?= mpicc
MPICXX ?= mpicxx
NVCC ?= nvcc
PYTHON ?= python3
CUDA_HOME ?= /usr/local/cuda

NX ?= 4096
NY ?= 4096
NSTEPS ?= 10000
TILE_WIDTH ?= 16
ARCH ?= sm_80
SCALING_NX_LIST ?= 1024 2048 4096
# 1 + (4900 / 100) = 50 snapshot times per scaling run (step 0..4900).
SCALING_STEPS ?= 4900
SCALING_SNAP_EVERY ?= 100
SCALING_MPI_NP ?= 2
SCALING_VARIANTS ?= serial omp cuda hybrid

COMMON_DEFS = -DNX_DEFAULT=$(NX) -DNY_DEFAULT=$(NY) -DNSTEPS_DEFAULT=$(NSTEPS)
OPT = -O3

CFLAGS = $(OPT) -std=c11 -Wall -Wextra $(COMMON_DEFS)
OMP_FLAGS = -fopenmp
NVCC_CUDA_FLAGS = $(OPT) $(COMMON_DEFS) -DUSE_CUDA -DTILE_WIDTH=$(TILE_WIDTH) -arch=$(ARCH)
MATH_LIBS = -lm
CUDA_INC = -I$(CUDA_HOME)/include
CUDA_LIB = -L$(CUDA_HOME)/lib64

EXECS = heat_serial heat_omp heat_cuda heat_hybrid

# Build all versions used in the report.
all: $(EXECS)

# ------------------------ Executable link targets -------------------------
heat_serial: main.serial.o stencil_cpu.serial.o
	$(CC) $(CFLAGS) $^ $(MATH_LIBS) -o $@

heat_omp: main.omp.o stencil_cpu.omp.o
	$(CC) $(CFLAGS) $(OMP_FLAGS) $^ $(MATH_LIBS) -o $@

heat_cuda: main.cuda.o stencil_cpu.cuda.o stencil_cuda.cuda.o
	$(NVCC) $(OPT) -arch=$(ARCH) $^ -Xcompiler "$(OMP_FLAGS)" $(MATH_LIBS) -o $@

heat_hybrid: main.hybrid.o stencil_cpu.hybrid.o halo.hybrid.o stencil_cuda.hybrid.o
	$(MPICXX) $(OPT) $(OMP_FLAGS) $^ $(CUDA_LIB) -lcudart $(MATH_LIBS) -o $@

# ------------------------- Compile rules: serial --------------------------
main.serial.o: main.c heat.h
	$(CC) $(CFLAGS) -DUSE_SERIAL -c $< -o $@

stencil_cpu.serial.o: stencil_cpu.c heat.h
	$(CC) $(CFLAGS) -DUSE_SERIAL -c $< -o $@

# -------------------------- Compile rules: OMP ----------------------------
main.omp.o: main.c heat.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -DUSE_OMP -c $< -o $@

stencil_cpu.omp.o: stencil_cpu.c heat.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -DUSE_OMP -c $< -o $@

# ------------------------- Compile rules: CUDA ----------------------------
main.cuda.o: main.c heat.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(CUDA_INC) -DUSE_OMP -DUSE_CUDA -c $< -o $@

stencil_cpu.cuda.o: stencil_cpu.c heat.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(CUDA_INC) -DUSE_OMP -DUSE_CUDA -c $< -o $@

stencil_cuda.cuda.o: stencil_cuda.cu heat.h
	$(NVCC) $(NVCC_CUDA_FLAGS) -c $< -o $@

# ---------------------- Compile rules: hybrid MPI+CUDA -------------------
main.hybrid.o: main.c heat.h
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) $(CUDA_INC) -DUSE_OMP -DUSE_CUDA -DUSE_MPI -c $< -o $@

stencil_cpu.hybrid.o: stencil_cpu.c heat.h
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) $(CUDA_INC) -DUSE_OMP -DUSE_CUDA -DUSE_MPI -c $< -o $@

halo.hybrid.o: halo.c heat.h
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) $(CUDA_INC) -DUSE_OMP -DUSE_CUDA -DUSE_MPI -c $< -o $@

stencil_cuda.hybrid.o: stencil_cuda.cu heat.h
	$(NVCC) $(NVCC_CUDA_FLAGS) -c $< -o $@

# Run the same scaling sweep used for performance plots.
# Choose subset with SCALING_VARIANTS (serial omp cuda hybrid).
scaling-study:
	@mkdir -p results
	@if [ ! -f results/scaling.csv ]; then \
		echo "version,nx,ny,nranks,nsteps,wall_time_s" > results/scaling.csv; \
	fi
	@for nx in $(SCALING_NX_LIST); do \
		for variant in $(SCALING_VARIANTS); do \
			case "$$variant" in \
				serial) \
					$(MAKE) --no-print-directory heat_serial; \
					./heat_serial --nx $$nx --ny $$nx --steps $(SCALING_STEPS) --snap-every $(SCALING_SNAP_EVERY) --outdir results/scaling_serial_nx$$nx >> results/scaling.csv; \
					;; \
				omp) \
					$(MAKE) --no-print-directory heat_omp; \
					./heat_omp --nx $$nx --ny $$nx --steps $(SCALING_STEPS) --snap-every $(SCALING_SNAP_EVERY) --outdir results/scaling_omp_nx$$nx >> results/scaling.csv; \
					;; \
				cuda) \
					$(MAKE) --no-print-directory heat_cuda; \
					./heat_cuda --nx $$nx --ny $$nx --steps $(SCALING_STEPS) --snap-every $(SCALING_SNAP_EVERY) --outdir results/scaling_cuda_nx$$nx >> results/scaling.csv; \
					;; \
				hybrid) \
					$(MAKE) --no-print-directory heat_hybrid; \
					mpirun --mca btl ^openib -np $(SCALING_MPI_NP) ./heat_hybrid --nx $$nx --ny $$nx --steps $(SCALING_STEPS) --snap-every $(SCALING_SNAP_EVERY) --outdir results/scaling_hybrid_nx$$nx >> results/scaling.csv; \
					;; \
				*) \
					echo "Unknown variant '$$variant'. Use serial/omp/cuda/hybrid."; \
					exit 1; \
					;; \
			esac; \
		done; \
	done
	$(PYTHON) visualize.py --mode scaling

# Remove generated binaries and object files.
clean:
	rm -f *.o $(EXECS)
