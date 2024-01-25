# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex
HEFFTE_HOME ?= ../../heffte/build


DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
USE_CUDA     = FALSE
USE_HIP      = FALSE
COMP         = gnu
DIM          = 3

USE_TIME_INTEGRATOR = FALSE
USE_SUNDIALS = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
VPATH_LOCATIONS += $(HEFFTE_HOME)/include
INCLUDE_LOCATIONS += $(HEFFTE_HOME)/include
LIBRARY_LOCATIONS += $(HEFFTE_HOME)/lib
INCLUDE_LOCATIONS += ../Source

libraries += -lheffte

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package 
include $(AMREX_HOME)/Tools/GNUMake/Make.rules

ifeq ($(USE_SUNDIALS),TRUE)
SUNDIALS_ROOT ?= ../../sundials/instdir
SUNDIALS_LIB_DIR ?= $(SUNDIALS_ROOT)/lib

USE_CVODE_LIBS ?= TRUE
USE_ARKODE_LIBS ?= TRUE

DEFINES += -DAMREX_USE_SUNDIALS
INCLUDE_LOCATIONS += $(SUNDIALS_ROOT)/include
LIBRARY_LOCATIONS += $(SUNDIALS_LIB_DIR)

LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_cvode
LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_arkode
LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_nvecmanyvector
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

