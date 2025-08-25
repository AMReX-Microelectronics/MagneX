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
option(MagneX_GPU_RDC "Enable relocatable device code" ON)

# Compute backend selection (matches FerroX strategy)
set(MagneX_COMPUTE_VALUES NOACC OMP CUDA HIP)
set(MagneX_COMPUTE NOACC CACHE STRING "On-node, accelerated computing backend (NOACC/OMP/CUDA/HIP)")
set_property(CACHE MagneX_COMPUTE PROPERTY STRINGS ${MagneX_COMPUTE_VALUES})
if(NOT MagneX_COMPUTE IN_LIST MagneX_COMPUTE_VALUES)
    message(FATAL_ERROR "MagneX_COMPUTE (${MagneX_COMPUTE}) must be one of ${MagneX_COMPUTE_VALUES}")
endif()

# Set dimensions (3D only for MagneX)
set(MagneX_DIMS 3 CACHE STRING "Simulation dimensionality (3D only)")

# AMReX dependency options (defined in cmake/dependencies/AMReX.cmake)
option(MagneX_amrex_internal "Download & build AMReX" ON)

# SUNDIALS integration support - matches FerroX SUNDIALS setup
# Default OFF matches GNU Make default behavior (only enabled when explicitly set)
if(MagneX_SUNDIALS)
    # Build/find SUNDIALS first, before AMReX looks for it
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/SUNDIALS.cmake)

    # Enable AMReX SUNDIALS support when MagneX SUNDIALS is requested
    set(AMReX_SUNDIALS ON CACHE BOOL "Enable SUNDIALS interfaces" FORCE)
endif()

# AMReX dependency - use advanced configuration from dependency file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/AMReX.cmake)

# Add after finding AMReX and SUNDIALS packages
if(MagneX_SUNDIALS AND AMReX_FOUND)
    message(STATUS "Checking AMReX/SUNDIALS type compatibility...")
    message(STATUS "DEBUG: FerroX_PRECISION=${FerroX_PRECISION}, SUNDIALS_PRECISION=${SUNDIALS_PRECISION}")

    file(WRITE ${CMAKE_BINARY_DIR}/test_types.cpp "#include <AMReX_REAL.H>\n#include <sundials/sundials_types.h>\n#include <type_traits>\n#include <iostream>\nint main() {\n    static_assert(sizeof(amrex::Real) == sizeof(sunrealtype));\n    static_assert(std::is_same<amrex::Real, sunrealtype>::value);\n    static_assert(sizeof(int) == sizeof(sunindextype));\n    std::cout << sizeof(amrex::Real) << 32 << sizeof(sunrealtype) << 10;\n    std::cout << sizeof(int) << 32 << sizeof(sunindextype) << 10;\n    return 0;\n}")

    try_compile(
        TYPES_COMPATIBLE
        ${CMAKE_BINARY_DIR}/test_build
        ${CMAKE_BINARY_DIR}/test_types.cpp
        CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${AMReX_INCLUDE_DIRS};${SUNDIALS_INCLUDE_DIR}"
        OUTPUT_VARIABLE COMPILE_OUTPUT
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

    # Handle SUNDIALS if enabled - matches FerroX advanced integration
    if(MagneX_SUNDIALS)
        target_compile_definitions(${binary_name} PRIVATE AMREX_USE_SUNDIALS)

        # Link SUNDIALS 7+ core component if available (following AMReX pattern)
        if(SUNDIALS_VERSION VERSION_GREATER_EQUAL 7)
            target_link_libraries(${binary_name} PRIVATE SUNDIALS::core)
        endif()

        target_link_libraries(${binary_name} PRIVATE
            SUNDIALS::cvode
            SUNDIALS::arkode
            SUNDIALS::nvecserial
            SUNDIALS::nvecmanyvector
        )

        if(MagneX_MPI)
            target_link_libraries(${binary_name} PUBLIC
                SUNDIALS::nvecmpimanyvector
            )
        endif()

        # Add backend-specific libraries
        if(MagneX_COMPUTE STREQUAL OMP)
            if(TARGET SUNDIALS::nvecopenmp)
                target_link_libraries(${binary_name} PRIVATE SUNDIALS::nvecopenmp)
            endif()
        elseif(MagneX_COMPUTE STREQUAL CUDA)
            if(TARGET SUNDIALS::nveccuda)
                target_link_libraries(${binary_name} PRIVATE SUNDIALS::nveccuda)
            endif()
            # Conditionally link fused CUDA libraries if available
            if(TARGET SUNDIALS::cvode_fused_cuda)
                target_link_libraries(${binary_name} PRIVATE SUNDIALS::cvode_fused_cuda)
            endif()
            # Also add ARKODE fused kernels if using ARKODE with CUDA
            if(TARGET SUNDIALS::arkode_fused_cuda)
                target_link_libraries(${binary_name} PRIVATE SUNDIALS::arkode_fused_cuda)
            endif()
        elseif(MagneX_COMPUTE STREQUAL HIP)
            if(TARGET SUNDIALS::nvechip)
                target_link_libraries(${binary_name} PRIVATE SUNDIALS::nvechip)
            endif()
            # Conditionally link fused HIP libraries if available
            if(TARGET SUNDIALS::cvode_fused_hip)
                target_link_libraries(${binary_name} PRIVATE SUNDIALS::cvode_fused_hip)
            endif()
            # Also add ARKODE fused kernels if using ARKODE with HIP
            if(TARGET SUNDIALS::arkode_fused_hip)
                target_link_libraries(${binary_name} PRIVATE SUNDIALS::arkode_fused_hip)
            endif()
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
