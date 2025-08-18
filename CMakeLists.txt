cmake_minimum_required(VERSION 3.24.0)
project(MagneX VERSION 1.0.0)

# In-source tree builds are not supported
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Change default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Options - mirror GNUmakefile defaults
option(MagneX_APP "Build the MagneX executable application" ON)
option(MagneX_MPI "Multi-node support (message-passing)" ON)
option(MagneX_FFT "FFT support" ON)
option(MagneX_SUNDIALS "SUNDIALS support" OFF)

# Compute backend selection (matches FerroX strategy)
set(MagneX_COMPUTE_VALUES NOACC OMP CUDA HIP)
set(MagneX_COMPUTE NOACC CACHE STRING "On-node, accelerated computing backend (NOACC/OMP/CUDA/HIP)")
set_property(CACHE MagneX_COMPUTE PROPERTY STRINGS ${MagneX_COMPUTE_VALUES})
if(NOT MagneX_COMPUTE IN_LIST MagneX_COMPUTE_VALUES)
    message(FATAL_ERROR "MagneX_COMPUTE (${MagneX_COMPUTE}) must be one of ${MagneX_COMPUTE_VALUES}")
endif()

# Set dimensions (3D only for MagneX)
set(MagneX_DIMS 3 CACHE STRING "Simulation dimensionality (3D only)")

# AMReX dependency options (similar to FerroX)
option(MagneX_amrex_internal "Download & build AMReX" ON)
set(MagneX_amrex_repo "https://github.com/AMReX-Codes/amrex.git" CACHE STRING "Repository URI to pull and build AMReX from")
set(MagneX_amrex_branch "development" CACHE STRING "Repository branch for MagneX_amrex_repo")
set(MagneX_amrex_src "" CACHE PATH "Path to AMReX source directory (if MagneX_amrex_internal=OFF)")

# SUNDIALS options (when enabled)
if(MagneX_SUNDIALS)
    option(MagneX_sundials_internal "Download & build SUNDIALS" ON)
    set(MagneX_sundials_src "" CACHE PATH "Path to SUNDIALS source directory (if MagneX_sundials_internal=OFF)")
endif()

# Set up AMReX dependency
if(MagneX_amrex_internal)
    # Download and build AMReX (like FerroX does)
    include(FetchContent)
    
    if(MagneX_amrex_src)
        # Use local source directory
        FetchContent_Declare(amrex
            SOURCE_DIR ${MagneX_amrex_src}
        )
    else()
        # Download from repository
        FetchContent_Declare(amrex
            GIT_REPOSITORY ${MagneX_amrex_repo}
            GIT_TAG ${MagneX_amrex_branch}
        )
    endif()
    
    # Configure AMReX options
    set(AMReX_SPACEDIM 3 CACHE STRING "" FORCE)
    set(AMReX_MPI ${MagneX_MPI} CACHE BOOL "" FORCE)
    set(AMReX_FFT ${MagneX_FFT} CACHE BOOL "" FORCE)
    if(MagneX_COMPUTE STREQUAL "OMP")
        set(AMReX_OMP ON CACHE BOOL "" FORCE)
    elseif(MagneX_COMPUTE STREQUAL "CUDA")
        set(AMReX_GPU_BACKEND CUDA CACHE STRING "" FORCE)
    elseif(MagneX_COMPUTE STREQUAL "HIP")
        set(AMReX_GPU_BACKEND HIP CACHE STRING "" FORCE)
    endif()
    if(MagneX_SUNDIALS)
        set(AMReX_SUNDIALS ON CACHE BOOL "" FORCE)
    endif()
    
    FetchContent_MakeAvailable(amrex)
else()
    # Find external AMReX installation
    find_package(AMReX REQUIRED
        COMPONENTS Base FFT Boundary
    )
endif()

# Create executable with sophisticated naming
if(MagneX_APP)
    # Build binary name like FerroX
    set(binary_name "main3d")
    
    # Add compiler info
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(binary_name "${binary_name}.gnu")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(binary_name "${binary_name}.intel")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(binary_name "${binary_name}.clang")
    endif()
    
    # Add MPI support
    if(MagneX_MPI)
        set(binary_name "${binary_name}.MPI")
    endif()
    
    # Add compute backend
    if(MagneX_COMPUTE STREQUAL "OMP")
        set(binary_name "${binary_name}.OMP")
    elseif(MagneX_COMPUTE STREQUAL "CUDA")
        set(binary_name "${binary_name}.CUDA")
    elseif(MagneX_COMPUTE STREQUAL "HIP")
        set(binary_name "${binary_name}.HIP")
    endif()
    
    set(binary_name "${binary_name}.ex")
    
    add_executable(${binary_name})
    
    # Add all source files from Make.package
    target_sources(${binary_name} PRIVATE
        Source/Checkpoint.cpp
        Source/ComputeLLGRHS.cpp
        Source/Demagnetization.cpp
        Source/Diagnostics.cpp
        Source/EffectiveAnisotropyField.cpp
        Source/EffectiveDMIField.cpp
        Source/EffectiveExchangeField.cpp
        Source/EvolveM_2nd.cpp
        Source/Initialization.cpp
        Source/MagneX.cpp
        Source/NormalizeM.cpp
        Source/main.cpp
        Source/Plotfile.cpp
    )
    
    # Include directories
    target_include_directories(${binary_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
    )
    
    # Link AMReX
    target_link_libraries(${binary_name} PRIVATE AMReX::amrex)
    
    # Handle SUNDIALS if enabled
    if(MagneX_SUNDIALS)
        target_compile_definitions(${binary_name} PRIVATE AMREX_USE_SUNDIALS)
        
        if(MagneX_sundials_internal)
            # Would need SUNDIALS FetchContent setup here
            message(STATUS "SUNDIALS internal build not implemented yet")
        else()
            # Find external SUNDIALS
            find_package(SUNDIALS REQUIRED)
            target_link_libraries(${binary_name} PRIVATE
                SUNDIALS::cvode
                SUNDIALS::arkode
                SUNDIALS::nvecmanyvector
                SUNDIALS::core
            )
        endif()
    endif()
    
    # Set output directory
    set_target_properties(${binary_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    # Create simple symlink for convenience
    add_custom_command(TARGET ${binary_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${binary_name}
        ${CMAKE_BINARY_DIR}/magnex
        COMMENT "Creating symlink magnex -> ${binary_name}"
    )
endif()
