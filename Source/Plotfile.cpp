#include "MagneX.H"

#include "AMReX_PlotFileUtil.H"

void WritePlotfile(MultiFab& Ms,
                   Array< MultiFab, AMREX_SPACEDIM>& Mfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_exchangefield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_DMIfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_anisotropyfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_demagfield,
                   MultiFab& PoissonRHS,
                   MultiFab& PoissonPhi,
                   const Geometry& geom,
                   const Real& time,
                   const int& plt_step)
{
    // timer for profiling
    BL_PROFILE_VAR("WritePlotfile()",WritePlotfile);

    BoxArray ba = Ms.boxArray();
    DistributionMapping dm = Ms.DistributionMap();
    
    MultiFab Plt(ba, dm, 21, 0);

    const std::string& pltfile = amrex::Concatenate("plt",plt_step,8);

    MultiFab::Copy(Plt, Ms, 0, 0, 1, 0);
    MultiFab::Copy(Plt, Mfield[0], 0, 1, 1, 0);
    MultiFab::Copy(Plt, Mfield[1], 0, 2, 1, 0);
    MultiFab::Copy(Plt, Mfield[2], 0, 3, 1, 0);
    MultiFab::Copy(Plt, H_biasfield[0], 0, 4, 1, 0);
    MultiFab::Copy(Plt, H_biasfield[1], 0, 5, 1, 0);
    MultiFab::Copy(Plt, H_biasfield[2], 0, 6, 1, 0);
    MultiFab::Copy(Plt, H_exchangefield[0], 0, 7, 1, 0);
    MultiFab::Copy(Plt, H_exchangefield[1], 0, 8, 1, 0);
    MultiFab::Copy(Plt, H_exchangefield[2], 0, 9, 1, 0);
    MultiFab::Copy(Plt, H_DMIfield[0], 0, 10, 1, 0);
    MultiFab::Copy(Plt, H_DMIfield[1], 0, 11, 1, 0);
    MultiFab::Copy(Plt, H_DMIfield[2], 0, 12, 1, 0);
    MultiFab::Copy(Plt, H_anisotropyfield[0], 0, 13, 1, 0);
    MultiFab::Copy(Plt, H_anisotropyfield[1], 0, 14, 1, 0);
    MultiFab::Copy(Plt, H_anisotropyfield[2], 0, 15, 1, 0);
    MultiFab::Copy(Plt, H_demagfield[0], 0, 16, 1, 0);
    MultiFab::Copy(Plt, H_demagfield[1], 0, 17, 1, 0);
    MultiFab::Copy(Plt, H_demagfield[2], 0, 18, 1, 0);
    MultiFab::Copy(Plt, PoissonRHS, 0, 19, 1, 0);
    MultiFab::Copy(Plt, PoissonPhi, 0, 20, 1, 0);

    WriteSingleLevelPlotfile(pltfile, Plt,
                             {"Ms",
                              "Mx",
                              "My",
                              "Mz",
                              "Hx_bias",
                              "Hy_bias",
                              "Hz_bias",
                              "Hx_exchange",
                              "Hy_exchange",
                              "Hz_exchange",
                              "Hx_DMI",
                              "Hy_DMI",
                              "Hz_DMI",
                              "Hx_anisotropy",
                              "Hy_anisotropy",
                              "Hz_anisotropy",
                              "Hx_demagfield","Hy_demagfield","Hz_demagfield",
                              "PoissonRHS","PoissonPhi"},
                             geom, time, plt_step);
}


