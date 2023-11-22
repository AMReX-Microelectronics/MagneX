#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void EvolveM_2nd(std::array< MultiFab, AMREX_SPACEDIM> &Mfield,
                 std::array< MultiFab, AMREX_SPACEDIM> &H_demagfield,
                 std::array< MultiFab, AMREX_SPACEDIM> &H_biasfield, // H bias
                 std::array< MultiFab, AMREX_SPACEDIM> &H_exchangefield, // effective exchange field
                 std::array< MultiFab, AMREX_SPACEDIM> &H_DMIfield,
                 std::array< MultiFab, AMREX_SPACEDIM> &H_anisotropyfield,
                 MultiFab                              &PoissonRHS, 
                 MultiFab                              &PoissonPhi, 
                 MultiFab                              &alpha,
                 MultiFab                              &Ms,
                 MultiFab                              &gamma,
                 MultiFab                              &exchange,
                 MultiFab                              &DMI,
                 MultiFab                              &anisotropy,
                 MultiFab                              &Kxx_dft_real,
                 MultiFab                              &Kxx_dft_imag,
                 MultiFab                              &Kxy_dft_real,
                 MultiFab                              &Kxy_dft_imag,
                 MultiFab                              &Kxz_dft_real,
                 MultiFab                              &Kxz_dft_imag,
                 MultiFab                              &Kyy_dft_real,
                 MultiFab                              &Kyy_dft_imag,
                 MultiFab                              &Kyz_dft_real,
                 MultiFab                              &Kyz_dft_imag,
                 MultiFab                              &Kzz_dft_real,
                 MultiFab                              &Kzz_dft_imag,
                 std::array< MultiFab, AMREX_SPACEDIM> &Mfield_padded,
                 GpuArray<int, 3>                      n_cell_large,
                 const Geometry&                       geom_large,
                 int demag_coupling,
                 int demag_solver,
                 int exchange_coupling,
                 int DMI_coupling,
                 int anisotropy_coupling,
                 amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
                 int M_normalization, 
                 Real mu0,
                 const Geometry& geom,
                 const Real& dt,
                 const Real& iterative_tolerance);
