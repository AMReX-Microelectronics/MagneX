# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
USE_CUDA     = FALSE
USE_HIP      = FALSE
COMP         = gnu
DIM          = 3
USE_FFT      = TRUE
USE_ML       = FALSE
TINY_PROFILE = FALSE
PROFILE      = FALSE
USE_SUNDIALS = FALSE
SUNDIALS_HOME ?= ../../sundials/instdir

ifeq ($(USE_ML),TRUE)

  CPPFLAGS += -DAMREX_USE_ML

  # Define a macro for the C++ preprocessor
  DEFINES += -DML_ENABLE -D_GLIBCXX_USE_CXX11_ABI=1

  # Pytorch root directory selection
  ifeq ($(USE_CUDA),TRUE)
    PYTORCH_ROOT := ../../libtorch_cuda
  else
    PYTORCH_ROOT := ../../libtorch_cpu
  endif

  TORCH_LIBPATH = $(PYTORCH_ROOT)/lib

  # Library definitions
  ifeq ($(USE_CUDA),TRUE)
    # Note: Modern LibTorch often requires both torch_cuda and torch_cpu
    TORCH_LIBS = -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcuda 
  else
    TORCH_LIBS = -ltorch -ltorch_cpu -lc10
  endif

  # Header search paths
  INCLUDE_LOCATIONS += $(PYTORCH_ROOT)/include \
                       $(PYTORCH_ROOT)/include/torch/csrc/api/include

  # Library search paths
  LIBRARY_LOCATIONS += $(TORCH_LIBPATH)

  # Linker flags (rpath ensures the .so files are found at runtime)
  ifeq ($(USE_CUDA),TRUE)
    LDFLAGS += -Xlinker "--no-as-needed,-rpath,$(TORCH_LIBPATH)" $(TORCH_LIBS)
  else
    LDFLAGS += -Wl,--no-as-needed,-rpath=$(TORCH_LIBPATH) $(TORCH_LIBS)
  endif
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/FFT/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package 
include $(AMREX_HOME)/Tools/GNUMake/Make.rules

