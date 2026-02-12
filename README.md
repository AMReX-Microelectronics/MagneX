# MagneX
MagneX is a massively parallel, 3D micromagnetics solver for modeling magnetic materials.
MagneX solves the Landau-Lifshitz-Gilbert (LLG) equations, including exchange, anisotropy, demagnetization, and Dzyaloshinskii-Moriya interaction (DMI) coupling.
The algorithm is implemented using Exascale Computing Project software framework, AMReX, which provides effective scalability on manycore and GPU-based supercomputing architectures.

For questions, please reach out to Zhi (Jackie) Yao (jackie_zhiyao@lbl.gov) and Andy Nonaka (ajnonaka@lbl.gov).

# Documentation and Getting Help
More extensive documentation is available [HERE](https://amrex-microelectronics.github.io).
Our community is here to help. Please report installation problems or general questions about the code in the github [Issues](https://github.com/AMReX-Microelectronics/MagneX/issues) tab above.
# Installation
Here are instructions for a basic, pure-MPI (no GPU) installation.  More detailed instructions for GPU systems are in the full documentation.
## Download AMReX and MagneX Repositories
Make sure that AMReX and MagneX are cloned at the same root location. \
``` >> git clone https://github.com/AMReX-Codes/amrex.git ``` \
``` >> git clone https://github.com/AMReX-Microelectronics/MagneX.git ```
## Dependencies
Beyond a standard Ubuntu22 installation, the Ubuntu packages libfftw3-dev, libfftw3-mpi-dev, and cmake are required.\
SUNDIALS is optional and enables Runge-Kutta, implicit, and multirate integrators (more detailed instructions in the full documentation).\
heFFTe is a required dependency.  At the same level that AMReX and MagneX are cloned, run: \
``` >> git clone https://github.com/icl-utk-edu/heffte.git ```\
``` >> cd heffte ```\
``` >> mkdir build ```\
``` >> cd build ```\
``` >> cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=. -DHeffte_ENABLE_FFTW=ON -DHeffte_ENABLE_CUDA=OFF .. ```\
``` >> make -j4 ```\
``` >> make install```
## Build

### GNU Make (Primary)
Navigate to MagneX/Exec/ and run:\
```>> make -j4```

### CMake (Alternative)
MagneX also supports building with CMake, which can automatically download and build dependencies.

#### Basic CMake Build
```bash
# CPU build with default options (NOACC backend)
cmake -S . -B build
cmake --build build -j 4

# OpenMP build  
cmake -S . -B build -DMagneX_COMPUTE=OMP
cmake --build build -j 4

# CUDA build
cmake -S . -B build -DMagneX_COMPUTE=CUDA
cmake --build build -j 4
```

#### Core Configuration Options
- **MagneX_COMPUTE**: `NOACC` (default), `OMP`, `CUDA`, `HIP` - Computing backend
- **MagneX_MPI**: `ON` (default), `OFF` - Multi-node support
- **MagneX_FFT**: `ON` (default), `OFF` - FFT support
- **MagneX_SUNDIALS**: `OFF` (default), `ON` - SUNDIALS ODE solver support

#### External Dependencies

**AMReX Configuration:**
```bash
# Use external AMReX installation
cmake -S . -B build \
  -DMagneX_amrex_internal=OFF \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX

# Use local AMReX source directory
cmake -S . -B build -DMagneX_amrex_src=/path/to/amrex/source

# Use custom AMReX repository/branch
cmake -S . -B build \
  -DMagneX_amrex_repo=https://github.com/user/amrex.git \
  -DMagneX_amrex_branch=my_branch

# Test with specific AMReX pull request (CI/testing)
cmake -S . -B build -DMagneX_amrex_pr=1234
```

**SUNDIALS Configuration (when MagneX_SUNDIALS=ON):**
```bash
# Use external SUNDIALS installation
cmake -S . -B build \
  -DMagneX_SUNDIALS=ON \
  -DMagneX_sundials_internal=OFF \
  -DSUNDIALS_DIR=/path/to/sundials/lib/cmake/sundials

# Use local SUNDIALS source directory
cmake -S . -B build \
  -DMagneX_SUNDIALS=ON \
  -DMagneX_sundials_src=/path/to/sundials/source
```

#### Example Build Commands
```bash
# Build with local AMReX source (recommended for development)
cmake -S . -B build -DMagneX_amrex_src=../amrex
cmake --build build -j 4

# OpenMP build with SUNDIALS support
cmake -S . -B build \
  -DMagneX_COMPUTE=OMP \
  -DMagneX_SUNDIALS=ON
cmake --build build -j 4

# CUDA build with external AMReX
export CMAKE_PREFIX_PATH=/path/to/amrex/install:$CMAKE_PREFIX_PATH
cmake -S . -B build \
  -DMagneX_COMPUTE=CUDA \
  -DMagneX_amrex_internal=OFF
cmake --build build -j 4
```

# Running MagneX

## GNU Make builds (from Exec directory)
You can run the following to simulate muMAG Standard Problem 4 dynamics:\
```>> ./main3d.gnu.MPI.ex standard_problem_inputs/inputs_std4```

## CMake builds (from project root directory)
```bash
# For CPU build (NOACC backend)
./build/main3d.gnu.MPI.ex Exec/standard_problem_inputs/inputs_std4

# For OpenMP build
./build/main3d.gnu.MPI.OMP.ex Exec/standard_problem_inputs/inputs_std4

# For CUDA build  
./build/main3d.gnu.MPI.CUDA.ex Exec/standard_problem_inputs/inputs_std4

# Using the convenience symlink
./build/magnex Exec/standard_problem_inputs/inputs_std4
```
# Visualization and Data Analysis
Refer to the following link for several visualization tools that can be used for AMReX plotfiles. 

[Visualization](https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html)

### Data Analysis in Python using yt 
You can extract the data in numpy array format using yt (you can refer to this for installation and usage of [yt](https://yt-project.org/). After you have installed yt, you can do something as follows, for example, to get variable 'Mx' (x-component of magnetization)
```
import yt
ds = yt.load('./plt00010000/') # for data at time step 10000
ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
Mx_array = ad0['Mx'].to_ndarray()
```
# Publications
1. Z. Yao, P. Kumar, J. C. LePelch, and A. Nonaka, MagneX: An Exascale-Enabled Micromagnetics Solver for Spintronic Systems, in preparation.
