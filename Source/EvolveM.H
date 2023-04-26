#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void Compute_LLG_RHS(//std::array< MultiFab, AMREX_SPACEDIM >&  LLG_RHS,
                   //std::array< MultiFab, AMREX_SPACEDIM >&   Mfield_old,
                   amrex::Vector<MultiFab>& LLG_RHS,
                   const amrex::Vector<MultiFab>& Mfield_old,
		   std::array< MultiFab, AMREX_SPACEDIM >&   H_demagfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   alpha,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
                   std::array< MultiFab, AMREX_SPACEDIM >&   gamma,
                   std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
                   std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
                   int demag_coupling,
                   int exchange_coupling,
                   int anisotropy_coupling,
                   amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
                   int M_normalization, 
                   Real mu0, 
                   const Geometry& geom,
		   const Real time);


void NormalizeM(amrex::Vector<MultiFab>& Mfield, 
		//std::array< MultiFab, AMREX_SPACEDIM >& Mfield,
		std::array< MultiFab, AMREX_SPACEDIM >& Ms, int M_normalization);

