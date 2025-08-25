macro(find_amrex)
    if(MagneX_amrex_src)
        message(STATUS "Compiling local AMReX ...")
        message(STATUS "AMReX source path: ${MagneX_amrex_src}")
        if(NOT IS_DIRECTORY ${MagneX_amrex_src})
            message(FATAL_ERROR "Specified directory MagneX_amrex_src='${MagneX_amrex_src}' does not exist!")
        endif()
    elseif(MagneX_amrex_internal)
        message(STATUS "Downloading AMReX ...")
        message(STATUS "AMReX repository: ${MagneX_amrex_repo} (${MagneX_amrex_branch})")
        include(FetchContent)
    endif()

    if(MagneX_amrex_internal OR MagneX_amrex_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options

        if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(AMReX_ASSERTIONS ON CACHE BOOL "")
            # note: floating-point exceptions can slow down debug runs a lot
            set(AMReX_FPE ON CACHE BOOL "")
        else()
            set(AMReX_ASSERTIONS OFF CACHE BOOL "")
            set(AMReX_FPE OFF CACHE BOOL "")
        endif()

        if(MagneX_COMPUTE STREQUAL OMP)
            set(AMReX_GPU_BACKEND  "NONE" CACHE INTERNAL "")
            set(AMReX_OMP          ON     CACHE INTERNAL "")
        elseif(MagneX_COMPUTE STREQUAL NOACC)
            set(AMReX_GPU_BACKEND  "NONE" CACHE INTERNAL "")
            set(AMReX_OMP          OFF    CACHE INTERNAL "")
        else()
            set(AMReX_GPU_BACKEND  "${MagneX_COMPUTE}" CACHE INTERNAL "")
            set(AMReX_OMP          OFF    CACHE INTERNAL "")
        endif()

        if(MagneX_FFT)
            set(AMReX_FFT ON CACHE INTERNAL "")
        else()
            set(AMReX_FFT OFF CACHE INTERNAL "")
        endif()

        if(MagneX_MPI)
            set(AMReX_MPI ON CACHE INTERNAL "")
        else()
            set(AMReX_MPI OFF CACHE INTERNAL "")
        endif()

        # MagneX uses DOUBLE precision by default
        set(AMReX_PRECISION "DOUBLE" CACHE INTERNAL "")
        set(AMReX_PARTICLES_PRECISION "DOUBLE" CACHE INTERNAL "")

        # MagneX-specific AMReX configuration
        set(AMReX_AMRLEVEL OFF CACHE INTERNAL "")
        set(AMReX_ENABLE_TESTS OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
        set(AMReX_BUILD_TUTORIALS OFF CACHE INTERNAL "")
        set(AMReX_PARTICLES ON CACHE INTERNAL "")
        set(AMReX_PROBINIT OFF CACHE INTERNAL "")
        set(AMReX_TINY_PROFILE ON CACHE BOOL "")
        set(AMReX_LINEAR_SOLVERS_EM ON CACHE INTERNAL "")
        set(AMReX_LINEAR_SOLVERS_INCFLO ON CACHE INTERNAL "")

        if(MagneX_GPU_RDC)
            set(AMReX_GPU_RDC ON CACHE BOOL "")
        else()
            # we don't need RDC and disabling it simplifies the build
            # complexity and potentially improves code optimization
            set(AMReX_GPU_RDC OFF CACHE BOOL "")
        endif()

        # Position independent code for shared libraries
        set(AMReX_PIC ON CACHE INTERNAL "" FORCE)

        # Install settings - static builds don't need install targets
        set(AMReX_INSTALL OFF CACHE INTERNAL "Generate Install Targets" FORCE)

        # MagneX is 3D only
        set(AMReX_SPACEDIM 3 CACHE INTERNAL "")

        if(MagneX_amrex_src)
            list(APPEND CMAKE_MODULE_PATH "${MagneX_amrex_src}/Tools/CMake")
            if(MagneX_COMPUTE STREQUAL CUDA)
                enable_language(CUDA)
                # AMReX 21.06+ supports CUDA_ARCHITECTURES
            elseif(MagneX_COMPUTE STREQUAL HIP)
                # HIP uses C++ compiler with special flags
                if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                    message(WARNING "HIP backend works best with Clang-based compilers (clang++, amdclang++, hipcc)")
                endif()
            elseif(MagneX_COMPUTE STREQUAL SYCL)
                # SYCL requires Intel oneAPI compiler
                if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
                    message(WARNING "SYCL backend requires Intel oneAPI compiler (icpx)")
                endif()
            endif()
            add_subdirectory(${MagneX_amrex_src} _deps/localamrex-build/)
        else()
            if(MagneX_COMPUTE STREQUAL CUDA)
                enable_language(CUDA)
                # AMReX 21.06+ supports CUDA_ARCHITECTURES
            elseif(MagneX_COMPUTE STREQUAL HIP)
                # HIP uses C++ compiler with special flags
                if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                    message(WARNING "HIP backend works best with Clang-based compilers (clang++, amdclang++, hipcc)")
                endif()
            elseif(MagneX_COMPUTE STREQUAL SYCL)
                # SYCL requires Intel oneAPI compiler
                if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
                    message(WARNING "SYCL backend requires Intel oneAPI compiler (icpx)")
                endif()
            endif()
            FetchContent_Declare(fetchedamrex
                GIT_REPOSITORY ${MagneX_amrex_repo}
                GIT_TAG        ${MagneX_amrex_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedamrex)
            list(APPEND CMAKE_MODULE_PATH "${fetchedamrex_SOURCE_DIR}/Tools/CMake")

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDAMREX)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDAMREX)
        endif()

        # AMReX options not relevant to most MagneX users
        mark_as_advanced(AMREX_BUILD_DATETIME)
        mark_as_advanced(AMReX_DIFFERENT_COMPILER)
        mark_as_advanced(AMReX_ENABLE_TESTS)
        mark_as_advanced(AMReX_SPACEDIM)
        mark_as_advanced(AMReX_AMRDATA)
        mark_as_advanced(AMReX_BASE_PROFILE) # mutually exclusive to tiny profile
        mark_as_advanced(AMReX_CONDUIT)
        mark_as_advanced(AMReX_CUDA)
        mark_as_advanced(AMReX_CUDA_COMPILATION_TIMER)
        mark_as_advanced(AMReX_CUDA_ERROR_CAPTURE_THIS)
        mark_as_advanced(AMReX_CUDA_ERROR_CROSS_EXECUTION_SPACE_CALL)
        mark_as_advanced(AMReX_CUDA_FASTMATH)
        mark_as_advanced(AMReX_CUDA_KEEP_FILES)
        mark_as_advanced(AMReX_CUDA_LTO)
        mark_as_advanced(AMReX_CUDA_MAXREGCOUNT)
        mark_as_advanced(AMReX_CUDA_MAX_THREADS)
        mark_as_advanced(AMReX_CUDA_PTX_VERBOSE)
        mark_as_advanced(AMReX_CUDA_SHOW_CODELINES)
        mark_as_advanced(AMReX_CUDA_SHOW_LINENUMBERS)
        mark_as_advanced(AMReX_CUDA_WARN_CAPTURE_THIS)
        mark_as_advanced(AMReX_GPU_RDC)
        mark_as_advanced(AMReX_PARTICLES)
        mark_as_advanced(AMReX_PARTICLES_PRECISION)
        mark_as_advanced(AMReX_DPCPP)
        mark_as_advanced(AMReX_EB)
        mark_as_advanced(AMReX_FPE)
        mark_as_advanced(AMReX_FORTRAN)
        mark_as_advanced(AMReX_FORTRAN_INTERFACES)
        mark_as_advanced(AMReX_HDF5)
        mark_as_advanced(AMReX_HIP)
        mark_as_advanced(AMReX_HYPRE)
        mark_as_advanced(AMReX_IPO)
        mark_as_advanced(AMReX_LINEAR_SOLVERS)
        mark_as_advanced(AMReX_LINEAR_SOLVERS_INCFLO)
        mark_as_advanced(AMReX_LINEAR_SOLVERS_EM)
        mark_as_advanced(AMReX_MEM_PROFILE)
        mark_as_advanced(AMReX_MPI)
        mark_as_advanced(AMReX_SIMD)
        mark_as_advanced(AMReX_OMP)
        mark_as_advanced(AMReX_PROBINIT)
        mark_as_advanced(AMReX_PETSC)
        mark_as_advanced(AMReX_PIC)
        mark_as_advanced(AMReX_SENSEI)
        mark_as_advanced(AMReX_SUNDIALS)
        mark_as_advanced(AMReX_TINY_PROFILE)
        mark_as_advanced(AMReX_TP_PROFILE)
        mark_as_advanced(USE_XSDK_DEFAULTS)

        message(STATUS "AMReX: Using version '${AMREX_PKG_VERSION}' (${AMREX_GIT_VERSION})")
    else()
        message(STATUS "Searching for pre-installed AMReX ...")
        # https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#importing-amrex-into-your-cmake-project

        if(MagneX_FFT)
            set(COMPONENT_FFT FFT)
        else()
            set(COMPONENT_FFT)
        endif()

        set(COMPONENT_PRECISION DOUBLE PDOUBLE)

        find_package(AMReX CONFIG REQUIRED COMPONENTS 3D ${COMPONENT_FFT} PARTICLES ${COMPONENT_PRECISION} LSOLVERS)

        # AMReX CMake helper scripts
        list(APPEND CMAKE_MODULE_PATH "${AMReX_DIR}/AMReXCMakeModules")

        message(STATUS "AMReX: Found version '${AMReX_VERSION}'")

        if(MagneX_COMPUTE STREQUAL CUDA)
            enable_language(CUDA)
        elseif(MagneX_COMPUTE STREQUAL HIP)
            # HIP uses C++ compiler with special flags
            if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                message(WARNING "HIP backend works best with Clang-based compilers (clang++, amdclang++, hipcc)")
            endif()
        elseif(MagneX_COMPUTE STREQUAL SYCL)
            # SYCL requires Intel oneAPI compiler
            if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
                message(WARNING "SYCL backend requires Intel oneAPI compiler (icpx)")
            endif()
        endif()
    endif()
endmacro()

# local source-tree
set(MagneX_amrex_src ""
    CACHE PATH
    "Local path to AMReX source directory (preferred if set)")

# Git fetcher
set(MagneX_amrex_repo "https://github.com/AMReX-Codes/amrex.git"
    CACHE STRING
    "Repository URI to pull and build AMReX from if(MagneX_amrex_internal)")

set(MagneX_amrex_branch "development"
    CACHE STRING
    "Repository branch for MagneX_amrex_repo if(MagneX_amrex_internal)")

find_amrex()