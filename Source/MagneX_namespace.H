namespace MagneX {

    //////////////////////////////////////////////////////
    // See MagneX.cpp for a description of these parameters
    //////////////////////////////////////////////////////
    
    extern AMREX_GPU_MANAGED amrex::GpuArray<int, 3> n_cell;

    extern int max_grid_size_x;
    extern int max_grid_size_y;
    extern int max_grid_size_z;

    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> prob_lo;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> prob_hi;

    extern int nsteps;

    extern amrex::Real stop_time;

    extern amrex::Real dt;

    extern int TimeIntegratorOption;

    extern amrex::Real fast_dt_ratio;
    extern int fast_H_bias;
    extern int fast_exchange;
    extern int fast_DMI;
    extern int fast_anisotropy;
    extern int fast_demag;

    extern amrex::Real iterative_tolerance;

    extern int plot_int;

    extern int plot_Ms;
    extern int plot_H_bias;
    extern int plot_exchange;
    extern int plot_DMI;
    extern int plot_anisotropy;
    extern int plot_demag;
    extern int plot_theta;

    extern int chk_int;

    extern int restart;

    extern int diag_type;

    extern int timedependent_Hbias;
    extern int timedependent_alpha;

    extern int Hbias_sweep;

    extern amrex::Real equilibrium_tolerance;

    extern AMREX_GPU_MANAGED amrex::Real increment_size;

    extern int nsteps_hysteresis;

    extern AMREX_GPU_MANAGED amrex::Real mu0;

    extern AMREX_GPU_MANAGED int M_normalization;
    extern AMREX_GPU_MANAGED int precession;
    extern AMREX_GPU_MANAGED int exchange_coupling;
    extern AMREX_GPU_MANAGED int DMI_coupling;
    extern AMREX_GPU_MANAGED int anisotropy_coupling;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> anisotropy_axis; 
    extern AMREX_GPU_MANAGED int demag_coupling;
    extern AMREX_GPU_MANAGED int FFT_solver;
}
