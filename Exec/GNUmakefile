# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
USE_CUDA     = TRUE
USE_HIP      = FALSE
COMP         = gnu
DIM          = 3

USE_SUNDIALS = TRUE

ifeq ($(USE_CUDA),TRUE)
  HEFFTE_HOME ?= ../../heffte/build_cuda
else ifeq ($(USE_CUDA),FALSE) 
  HEFFTE_HOME ?= ../../heffte/build
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
VPATH_LOCATIONS += $(HEFFTE_HOME)/include
INCLUDE_LOCATIONS += $(HEFFTE_HOME)/include
LIBRARY_LOCATIONS += $(HEFFTE_HOME)/lib
INCLUDE_LOCATIONS += ../Source

libraries += -lheffte

ifeq ($(USE_SUNDIALS),TRUE)
  SUNDIALS_ROOT ?= $(TOP)../../sundials-6.7.0/instdir_cuda
  SUNDIALS_LIB_DIR ?= $(SUNDIALS_ROOT)/lib

  USE_CVODE_LIBS ?= TRUE
  USE_ARKODE_LIBS ?= TRUE

  DEFINES += -DAMREX_USE_SUNDIALS
  INCLUDE_LOCATIONS += $(SUNDIALS_ROOT)/include
  LIBRARY_LOCATIONS += $(SUNDIALS_LIB_DIR)

  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_cvode
  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_arkode
  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_nvecmanyvector
  
  ifeq ($(USE_CUDA),TRUE)
  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_nveccuda
  endif

endif

ifeq ($(USE_CUDA),TRUE)
  libraries += -lcufft
else ifeq ($(USE_HIP),TRUE)
  # Use rocFFT.  ROC_PATH is defined in amrex
  INCLUDE_LOCATIONS += $(ROC_PATH)/rocfft/include
  LIBRARY_LOCATIONS += $(ROC_PATH)/rocfft/lib
  LIBRARIES += -L$(ROC_PATH)/rocfft/lib -lrocfft
else
  libraries += -lfftw3_mpi -lfftw3f -lfftw3
endif

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package 
include $(AMREX_HOME)/Tools/GNUMake/Make.rules

