#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;
void mf_avg_fc_to_cc(MultiFab&  Plt,
                   //std::array< MultiFab, AMREX_SPACEDIM >&   Mfield,
		   amrex::Vector<MultiFab>& Mfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms);
