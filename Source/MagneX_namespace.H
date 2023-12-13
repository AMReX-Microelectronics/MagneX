namespace MagneX {

    // Number of cells in each dimension
    extern amrex::GpuArray<int, 3> n_cell;

    extern int max_grid_size;

    // physical lo/hi coordiates
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> prob_lo;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> prob_hi;

    // total steps in simulation
    extern int nsteps;

    // 1 = first order forward Euler
    // 2 = iterative predictor-corrector
    // 3 = iterative direct solver
    // 4 = AMReX and SUNDIALS integrators
    extern int TimeIntegratorOption;

    // tolerance threhold (L_inf change between iterations) for TimeIntegrationOption 2 and 3
    // for TimeIntegrationOption=2, iterative_tolerance=0. means force 2 iterations
    // for TimeIntegrationOption=3, iterative_tolerance=0. means force 1 iteration
    extern amrex::Real iterative_tolerance;

    // time step
    extern amrex::Real dt;

    // how often to write a plotfile
    extern int plot_int;

    // how often to write a checkpoextern int
    extern int chk_int;

    // step to restart from
    extern int restart;

    // what type of extra diagnostics?
    // 4 = standard problem 4
    extern int diag_type;

    // permeability
    extern amrex::Real mu0;

    // whether to call the parser each time step, or only at initialization
    extern int timedependent_Hbias;
    extern int timedependent_alpha;
    
    // turn on demagnetization
    extern int demag_coupling;

    // demagnetization solver type
    // -1 = periodic/Neumann MLMG
    // 0 = Open Poisson MLMG
    // 1 = FFT-based
    extern int demag_solver;

    // 0 = unsaturated; 1 = saturated
    extern int M_normalization;

    // turn on exchange
    extern int exchange_coupling;

    // turn on DMI
    extern int DMI_coupling;

    // turn on anisotropy
    extern int anisotropy_coupling;
    extern amrex::GpuArray<amrex::Real, 3> anisotropy_axis; 

}
