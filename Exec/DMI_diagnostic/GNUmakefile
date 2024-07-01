AMREX_HOME ?= ../../../amrex

DEBUG	= TRUE
DIM	= 3

COMP    = gcc

PRECISION = DOUBLE

USE_MPI   = FALSE
USE_OMP   = FALSE

###################################################

EBASE = DMI_diagnostic

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

vpath %.c   : . $(vpathdir)
vpath %.h   : . $(vpathdir)
vpath %.cpp : . $(vpathdir)
vpath %.H   : . $(vpathdir)
vpath %.F   : . $(vpathdir)
vpath %.f   : . $(vpathdir)
vpath %.f90 : . $(vpathdir)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

clean::
	$(SILENT) $(RM) particle_compare.exe
