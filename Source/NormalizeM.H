#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void NormalizeM( Array< MultiFab, AMREX_SPACEDIM >& Mfield,
		 MultiFab& Ms, int M_normalization,
                const Geometry& geom);
