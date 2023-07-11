# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = TRUE
USE_CUDA     = FALSE
COMP         = gnu
DIM          = 3

NEUMANN = FALSE
USE_TIME_INTEGRATOR = FALSE
USE_SUNDIALS = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package 
include $(AMREX_HOME)/Src/LinearSolvers/MLMG/Make.package 
include $(AMREX_HOME)/Src/LinearSolvers/OpenBC/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules

ifeq ($(NEUMANN), TRUE)
  DEFINES += -DNEUMANN
endif

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

