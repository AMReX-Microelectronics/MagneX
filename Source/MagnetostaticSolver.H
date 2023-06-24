#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void ComputePoissonRHS(MultiFab&                        PoissonRHS,
                       //Array<MultiFab, AMREX_SPACEDIM>& Mfield,
		       const amrex::Vector<MultiFab>& Mfield,
                       Array<MultiFab, AMREX_SPACEDIM>& Ms,
                       const Geometry&                 geom);

void ComputeHfromPhi(MultiFab&                         PoissonPhi,
                      Array<MultiFab, AMREX_SPACEDIM>& H_demagfield,
                      amrex::GpuArray<amrex::Real, 3> prob_lo,
                      amrex::GpuArray<amrex::Real, 3> prob_hi,
                      const Geometry&                 geom);
