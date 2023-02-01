#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void WriteCheckPoint(int step,
                     const amrex::Real time,
                     std::array< MultiFab, AMREX_SPACEDIM>& Mfield,
                     std::array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
		     std::array< MultiFab, AMREX_SPACEDIM>& H_demagfield);

void ReadCheckPoint(int& restart,
		    amrex::Real& time,
		    std::array< MultiFab, AMREX_SPACEDIM>& Mfield,
		    std::array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
		    std::array< MultiFab, AMREX_SPACEDIM>& H_demagfield,
		    BoxArray& ba,
		    DistributionMapping& dm);
