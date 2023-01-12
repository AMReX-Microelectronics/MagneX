#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void WriteCheckPoint(int step,
                     const amrex::Real time,
                     std::array< MultiFab, AMREX_SPACEDIM>& Mfiel,
                     std::array< MultiFab, AMREX_SPACEDIM>& H_biasfieldd);

void ReadCheckPoint(int& step,
		    int& restart,
		    amrex::Real& time,
		    std::array< MultiFab, AMREX_SPACEDIM>& Mfield,
		    std::array< MultiFab, AMREX_SPACEDIM>& H_biasfield);
