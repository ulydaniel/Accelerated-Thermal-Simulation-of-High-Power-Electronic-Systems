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

NX ?= 4096
NY ?= 4096
NSTEPS ?= 10000
TILE_WIDTH ?= 16
ARCH ?= sm_80

COMMON_DEFS = -DNX_DEFAULT=$(NX) -DNY_DEFAULT=$(NY) -DNSTEPS_DEFAULT=$(NSTEPS)
OPT = -O3

CFLAGS = $(OPT) -std=c11 -Wall -Wextra $(COMMON_DEFS)
OMP_FLAGS = -fopenmp
NVCC_CUDA_FLAGS = $(OPT) $(COMMON_DEFS) -DUSE_CUDA -DTILE_WIDTH=$(TILE_WIDTH) -arch=$(ARCH)

EXECS = heat_serial heat_omp heat_cuda heat_hybrid

# Build all versions used in the report.
all: $(EXECS)

# ------------------------ Executable link targets -------------------------
heat_serial: main.serial.o stencil_cpu.serial.o
	$(CC) $(CFLAGS) $^ -o $@

heat_omp: main.omp.o stencil_cpu.omp.o
	$(CC) $(CFLAGS) $(OMP_FLAGS) $^ -o $@

heat_cuda: main.cuda.o stencil_cpu.cuda.o stencil_cuda.cuda.o
	$(NVCC) $(OPT) -arch=$(ARCH) $^ -Xcompiler "$(OMP_FLAGS)" -o $@

heat_hybrid: main.hybrid.o stencil_cpu.hybrid.o halo.hybrid.o stencil_cuda.hybrid.o
	$(MPICXX) $(OPT) $(OMP_FLAGS) $^ -lcudart -o $@

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
	$(CC) $(CFLAGS) $(OMP_FLAGS) -DUSE_OMP -DUSE_CUDA -c $< -o $@

stencil_cpu.cuda.o: stencil_cpu.c heat.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -DUSE_OMP -DUSE_CUDA -c $< -o $@

stencil_cuda.cuda.o: stencil_cuda.cu heat.h
	$(NVCC) $(NVCC_CUDA_FLAGS) -c $< -o $@

# ---------------------- Compile rules: hybrid MPI+CUDA -------------------
main.hybrid.o: main.c heat.h
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) -DUSE_OMP -DUSE_CUDA -DUSE_MPI -c $< -o $@

stencil_cpu.hybrid.o: stencil_cpu.c heat.h
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) -DUSE_OMP -DUSE_CUDA -DUSE_MPI -c $< -o $@

halo.hybrid.o: halo.c heat.h
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) -DUSE_OMP -DUSE_CUDA -DUSE_MPI -c $< -o $@

stencil_cuda.hybrid.o: stencil_cuda.cu heat.h
	$(NVCC) $(NVCC_CUDA_FLAGS) -c $< -o $@

# Run the same scaling sweep used for performance plots.
scaling-study:
	@mkdir -p results
	@if [ ! -f results/scaling.csv ]; then \
		echo "variant,nx,ny,nsteps,tile_width,arch,time_s,speedup" > results/scaling.csv; \
	fi
	@for nx in 1024 2048 4096; do \
		$(MAKE) --no-print-directory NX=$$nx NY=$$nx heat_serial; \
		./heat_serial >> results/scaling.csv; \
		$(MAKE) --no-print-directory NX=$$nx NY=$$nx heat_omp; \
		./heat_omp >> results/scaling.csv; \
		$(MAKE) --no-print-directory NX=$$nx NY=$$nx TILE_WIDTH=$(TILE_WIDTH) ARCH=$(ARCH) heat_cuda; \
		./heat_cuda >> results/scaling.csv; \
		$(MAKE) --no-print-directory NX=$$nx NY=$$nx TILE_WIDTH=$(TILE_WIDTH) ARCH=$(ARCH) heat_hybrid; \
		mpirun -np 2 ./heat_hybrid >> results/scaling.csv; \
	done
	$(PYTHON) visualize.py --mode scaling

# Remove generated binaries and object files.
clean:
	rm -f *.o $(EXECS)
