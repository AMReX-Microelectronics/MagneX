# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
USE_CUDA     = FALSE
USE_HIP      = FALSE
COMP         = gnu
DIM          = 3

USE_SUNDIALS = FALSE

ifeq ($(USE_CUDA),TRUE)
  HEFFTE_HOME ?= ../../heffte/build
else ifeq ($(USE_CUDA),FALSE) 
  HEFFTE_HOME ?= ../../heffte/build
endif


# Pytorch directories 
ifeq ($(USE_CUDA),TRUE)
  PYTORCH_ROOT := ../libtorch_cuda
else
  PYTORCH_ROOT := ../libtorch_cpu
endif
TORCH_LIBPATH = $(PYTORCH_ROOT)/lib

ifeq ($(USE_CUDA),TRUE)
  TORCH_LIBS = -ltorch -ltorch_cpu -lc10 -lc10_cuda -lcuda 
else
  TORCH_LIBS = -ltorch -ltorch_cpu -lc10
endif

INCLUDE_LOCATIONS += $(PYTORCH_ROOT)/include \
                     $(PYTORCH_ROOT)/include/torch/csrc/api/include
LIBRARY_LOCATIONS += $(TORCH_LIBPATH)

DEFINES += -D_GLIBCXX_USE_CXX11_ABI=1
ifeq ($(USE_CUDA),TRUE)
  LDFLAGS += -Xlinker "--no-as-needed,-rpath $(TORCH_LIBPATH) $(TORCH_LIBS)"
else
  LDFLAGS += -Wl,--no-as-needed,-rpath=$(TORCH_LIBPATH) $(TORCH_LIBS)
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
  ifeq ($(USE_CUDA),TRUE)
    SUNDIALS_ROOT ?= $(TOP)../../sundials/instdir_cuda
  else
    SUNDIALS_ROOT ?= $(TOP)../../sundials/instdir
  endif
  SUNDIALS_LIB_DIR ?= $(SUNDIALS_ROOT)/lib

  USE_CVODE_LIBS ?= TRUE
  USE_ARKODE_LIBS ?= TRUE

  DEFINES += -DAMREX_USE_SUNDIALS
  INCLUDE_LOCATIONS += $(SUNDIALS_ROOT)/include
  LIBRARY_LOCATIONS += $(SUNDIALS_LIB_DIR)

  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_cvode
  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_arkode
  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_nvecmanyvector
  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_core  # for newer versions of sundials
  
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
