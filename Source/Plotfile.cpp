#include "MagneX.H"

#include "AMReX_PlotFileUtil.H"

void WritePlotfile(MultiFab& Ms,
                   Array< MultiFab, AMREX_SPACEDIM>& Mfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_exchangefield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_DMIfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_anisotropyfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_demagfield,
                   const Geometry& geom,
                   const Real& time,
                   const int& plt_step)
{
    // timer for profiling
    BL_PROFILE_VAR("WritePlotfile()",WritePlotfile);

    BoxArray ba = Ms.boxArray();
    DistributionMapping dm = Ms.DistributionMap();

    const std::string& pltfile = amrex::Concatenate("plt",plt_step,8);

    Vector<std::string> var_names;

    // Mx, My, Mz
    int nvar = 3;

    if (plot_Ms) {
        ++nvar;
        var_names.push_back("Ms");
    }
    var_names.push_back("Mx");
    var_names.push_back("My");
    var_names.push_back("Mz");

    if (plot_H_bias) {
        nvar += 3;
        var_names.push_back("Hx_bias");
        var_names.push_back("Hy_bias");
        var_names.push_back("Hz_bias");
    }

    if (plot_exchange && exchange_coupling) {
        nvar += 3;
        var_names.push_back("Hx_exchange");
        var_names.push_back("Hy_exchange");
        var_names.push_back("Hz_exchange");
    }

    if (plot_DMI && DMI_coupling) {
        nvar += 3;
        var_names.push_back("Hx_DMI");
        var_names.push_back("Hy_DMI");
        var_names.push_back("Hz_DMI");
    }

    if (plot_anisotropy && anisotropy_coupling) {
        nvar += 3;
        var_names.push_back("Hx_anisotropy");
        var_names.push_back("Hy_anisotropy");
        var_names.push_back("Hz_anisotropy");
    }

    if (plot_demag && demag_coupling) {
        nvar += 3;
        var_names.push_back("Hx_demagfield");
        var_names.push_back("Hy_demagfield");
        var_names.push_back("Hz_demagfield");
    }

    MultiFab Plt(ba, dm, nvar, 0);

    int counter = 0;
    
    if (plot_Ms) {
        MultiFab::Copy(Plt, Ms, 0, counter++, 1, 0);
    }
    MultiFab::Copy(Plt, Mfield[0], 0, counter++, 1, 0);
    MultiFab::Copy(Plt, Mfield[1], 0, counter++, 1, 0);
    MultiFab::Copy(Plt, Mfield[2], 0, counter++, 1, 0);
    if (plot_H_bias) {
        MultiFab::Copy(Plt, H_biasfield[0], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_biasfield[1], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_biasfield[2], 0, counter++, 1, 0);
    }
    if (plot_exchange && exchange_coupling) {
        MultiFab::Copy(Plt, H_exchangefield[0], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_exchangefield[1], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_exchangefield[2], 0, counter++, 1, 0);
    }
    if (plot_DMI && DMI_coupling) {
        MultiFab::Copy(Plt, H_DMIfield[0], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_DMIfield[1], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_DMIfield[2], 0, counter++, 1, 0);
    }
    if (plot_anisotropy && anisotropy_coupling) {
        MultiFab::Copy(Plt, H_anisotropyfield[0], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_anisotropyfield[1], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_anisotropyfield[2], 0, counter++, 1, 0);
    }
    if (plot_demag && demag_coupling) {
        MultiFab::Copy(Plt, H_demagfield[0], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[1], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[2], 0, counter++, 1, 0);
    }

    WriteSingleLevelPlotfile(pltfile, Plt, var_names, geom, time, plt_step);
}


