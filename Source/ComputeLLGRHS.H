#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;
void Compute_LLG_RHS(
                   Array< MultiFab, AMREX_SPACEDIM >&  LLG_RHS,
                   const Array< MultiFab, AMREX_SPACEDIM >&   Mfield_old,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_demagfield,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_exchangefield,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_DMIfield,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_anisotropyfield,
                   MultiFab&   alpha,
                   MultiFab&   Ms,
                   MultiFab&   gamma,
                   int demag_coupling,
                   int exchange_coupling,
                   int DMI_coupling,
                   int anisotropy_coupling,
                   int M_normalization,
                   Real mu0);
