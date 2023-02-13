
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 
#include <AMReX_OpenBC.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include "myfunc.H"
#include "Initialization.H"
#include "MagnetostaticSolver.H"
#include "MagLaplacian.H"
#include "Diagnostics.H"
#include "EvolveM.H"
#include "EvolveM_2nd.H"
#include "Checkpoint.H"
#include <AMReX_TimeIntegrator.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{

    Real total_step_strt_time = ParallelDescriptor::second();

    // **********************************
    // SIMULATION PARAMETERS

    amrex::GpuArray<int, 3> n_cell; // Number of cells in each dimension

    // size of each box (or grid)
    int max_grid_size;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // ho often to write a checkpoint
    int chk_int;

    // step to restart from
    int restart;

    // time step
    Real dt;
    
    amrex::GpuArray<amrex::Real, 3> prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, 3> prob_hi; // physical hi coordinate

    amrex::GpuArray<amrex::Real, 3> mag_lo; // physical lo coordinate of magnetic region
    amrex::GpuArray<amrex::Real, 3> mag_hi; // physical hi coordinate of magnetic region

    int TimeIntegratorOrder;

    // Magnetic Properties
    Real alpha_val, gamma_val, Ms_val, exchange_val, anisotropy_val;
    Real mu0;
    amrex::GpuArray<amrex::Real, 3> anisotropy_axis; 

    int demag_coupling;
    int M_normalization;
    int exchange_coupling;
    int anisotropy_coupling;


    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        amrex::Vector<int> temp_int(AMREX_SPACEDIM);
        if (pp.queryarr("n_cell",temp_int)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                n_cell[i] = temp_int[i];
            }
        }

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        pp.get("TimeIntegratorOrder",TimeIntegratorOrder);

        // Material Properties
	
        pp.get("mu0",mu0);
        pp.get("alpha_val",alpha_val);
        pp.get("gamma_val",gamma_val);
        pp.get("Ms_val",Ms_val);
        pp.get("exchange_val",exchange_val);
        pp.get("anisotropy_val",anisotropy_val);

        pp.get("demag_coupling",demag_coupling);
        pp.get("M_normalization", M_normalization);
        pp.get("exchange_coupling", exchange_coupling);
        pp.get("anisotropy_coupling", anisotropy_coupling);


        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

	// Default chk_int to -1, allow us to set it to something else in the inputs file
        //  If chk_int < 0 then no chk files will be written
        chk_int = -1;
        pp.query("chk_int",chk_int);

	// query restart
	restart = -1;
	pp.query("restart",restart);
	
        // time step
        pp.get("dt",dt);

        amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);
        if (pp.queryarr("prob_lo",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                prob_lo[i] = temp[i];
            }
        }
        if (pp.queryarr("prob_hi",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                prob_hi[i] = temp[i];
            }
        }

        if (pp.queryarr("mag_lo",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                mag_lo[i] = temp[i];
            }
        }
        if (pp.queryarr("mag_hi",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                mag_hi[i] = temp[i];
            }
        }
        if (pp.queryarr("anisotropy_axis",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                anisotropy_axis[i] = temp[i];
            }
        }
    }

    int start_step = 1;

    // time = starting time in the simulation
    Real time = 0.0;	

    //Array<MultiFab, AMREX_SPACEDIM> Mfield;
    Array<MultiFab, AMREX_SPACEDIM> H_biasfield;
    Array<MultiFab, AMREX_SPACEDIM> H_demagfield;

    amrex::Vector<MultiFab> Mfield(AMREX_SPACEDIM);

    BoxArray ba;
    DistributionMapping dm;
    
    if (restart > 0) {

      start_step = restart+1;

      // read in Mfield, H_biasfield, and ba
      // create a DistributionMapping dm
      ReadCheckPoint(restart,time,Mfield,H_biasfield,H_demagfield,ba,dm);
      
    }
    
    // **********************************
    // SIMULATION SETUP

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);
    
    if (restart == -1) {
      // Initialize the boxarray "ba" from the single box "domain"
      ba.define(domain);

      // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
      ba.maxSize(max_grid_size);
    }

    // This defines the physical box in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});

    // periodic in x and y directions
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)}; // nonperiodic in all directions

    // This defines a Geometry object
    Geometry geom;
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 2;

    // How Boxes are distrubuted among MPI processes
    if (restart == -1) {
      dm.define(ba);
    }

    // Allocate multifabs
    if (restart == -1) {
      // face-centered Mfield
      AMREX_D_TERM(Mfield[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
		   Mfield[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
		   Mfield[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost););

      // face-centered H_biasfield
      AMREX_D_TERM(H_biasfield[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, 0);,
		   H_biasfield[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, 0);,
		   H_biasfield[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, 0););

      for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        H_demagfield[dir].define(ba, dm, 1, 1);
      }
    }

    //Array<MultiFab, AMREX_SPACEDIM> Mfield_old;
    amrex::Vector<MultiFab> Mfield_old(AMREX_SPACEDIM);
    // face-centered Mfield_old
    AMREX_D_TERM(Mfield_old[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
                 Mfield_old[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
                 Mfield_old[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost););

    //Array<MultiFab, AMREX_SPACEDIM> Mfield_prev_iter;
    amrex::Vector<MultiFab> Mfield_prev_iter(AMREX_SPACEDIM);
    // face-centered Mfield at predictor step (for 2nd order time integrator)
    AMREX_D_TERM(Mfield_prev_iter[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
                 Mfield_prev_iter[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
                 Mfield_prev_iter[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost););

    //Array<MultiFab, AMREX_SPACEDIM> Mfield_error;
    amrex::Vector<MultiFab> Mfield_error(AMREX_SPACEDIM);
    // face-centered Mfield at predictor step (for 2nd order time integrator)
    AMREX_D_TERM(Mfield_error[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, 0);,
                 Mfield_error[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, 0);,
                 Mfield_error[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, 0););

    //Array<MultiFab, AMREX_SPACEDIM> LLG_RHS;
    amrex::Vector<MultiFab> LLG_RHS(AMREX_SPACEDIM);
    // face-centered LLG_RHS
    AMREX_D_TERM(LLG_RHS[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, 0);,
                 LLG_RHS[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, 0);,
                 LLG_RHS[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, 0););

    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS_pre;
    // face-centered LLG_RHS
    AMREX_D_TERM(LLG_RHS_pre[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, 0);,
                 LLG_RHS_pre[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, 0);,
                 LLG_RHS_pre[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, 0););

    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS_avg;
    // face-centered LLG_RHS
    AMREX_D_TERM(LLG_RHS_avg[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, 0);,
                 LLG_RHS_avg[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, 0);,
                 LLG_RHS_avg[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, 0););

    //Face-centered magnetic properties
    std::array< MultiFab, AMREX_SPACEDIM > alpha;
    AMREX_D_TERM(alpha[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 alpha[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 alpha[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););

    std::array< MultiFab, AMREX_SPACEDIM > gamma;
    AMREX_D_TERM(gamma[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 gamma[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 gamma[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););

    std::array< MultiFab, AMREX_SPACEDIM > Ms;
    AMREX_D_TERM(Ms[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 1);,
                 Ms[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 1);,
                 Ms[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 1););

    std::array< MultiFab, AMREX_SPACEDIM > exchange;
    AMREX_D_TERM(exchange[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 exchange[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 exchange[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););

    std::array< MultiFab, AMREX_SPACEDIM > anisotropy;
    AMREX_D_TERM(anisotropy[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 anisotropy[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 anisotropy[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););


    amrex::Print() << "==================== Initial Setup ====================\n";
    amrex::Print() << " demag_coupling      = " << demag_coupling      << "\n";
    amrex::Print() << " M_normalization     = " << M_normalization     << "\n";
    amrex::Print() << " exchange_coupling   = " << exchange_coupling   << "\n";
    amrex::Print() << " anisotropy_coupling = " << anisotropy_coupling << "\n";
    amrex::Print() << " Ms                  = " << Ms_val              << "\n";
    amrex::Print() << " alpha               = " << alpha_val           << "\n";
    amrex::Print() << " gamma               = " << gamma_val           << "\n";
    amrex::Print() << " exchange_value      = " << exchange_val        << "\n";
    amrex::Print() << " anisotropy_value    = " << anisotropy_val      << "\n";
    amrex::Print() << "=======================================================\n";

    MultiFab PoissonRHS(ba, dm, 1, 0);
    MultiFab PoissonPhi(ba, dm, 1, 1); // one ghost cell

    MultiFab Plt(ba, dm, 26, 0);

    //Solver for Poisson equation
    LPInfo info;
#ifdef NEUMANN
    MLABecLaplacian mlabec({geom}, {ba}, {dm}, info);

    mlabec.setEnforceSingularSolvable(false);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    std::array<LinOpBCType, AMREX_SPACEDIM> lo_mlmg_bc;
    std::array<LinOpBCType, AMREX_SPACEDIM> hi_mlmg_bc;

    //Periodic
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if(is_periodic[idim]){
          lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Periodic;
        } else {
          lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Neumann;
        }
    }

    mlabec.setDomainBC(lo_mlmg_bc,hi_mlmg_bc);

    { // add this brace so alpha_cc and beta_face go out of scope afterwards (save memory)
        // coefficients for solver
        MultiFab alpha_cc(ba, dm, 1, 0);
        std::array< MultiFab, AMREX_SPACEDIM > beta_face;
        AMREX_D_TERM(beta_face[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                     beta_face[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                     beta_face[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););

        alpha_cc.setVal(0.);
        beta_face[0].setVal(1.);
        beta_face[1].setVal(1.);
        beta_face[2].setVal(1.);

        // (A*alpha_cc - B * div beta grad) phi = rhs
        mlabec.setScalars(0.0, 1.0); // A = 0.0, B = 1.0
        mlabec.setACoeffs(0, alpha_cc); //First argument 0 is lev
        mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(beta_face));
    }

    //Declare MLMG object
    MLMG mlmg(mlabec);
    mlmg.setVerbose(2);
#else 
    OpenBCSolver openbc({geom}, {ba}, {dm}, info);
    openbc.setVerbose(2);
#endif

    InitializeMagneticProperties(alpha, Ms, gamma, exchange, anisotropy,
                                 alpha_val, Ms_val, gamma_val, exchange_val, anisotropy_val, 
                                 prob_lo, prob_hi, mag_lo, mag_hi, geom);

    if (restart == -1) {      
      //Initialize fields
      InitializeFields(Mfield, H_biasfield, Ms, prob_lo, prob_hi, geom);

      if(demag_coupling == 1){ 
        //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
        //Compute RHS of Poisson equation
        ComputePoissonRHS(PoissonRHS, Mfield, Ms, geom);
        
        //Initial guess for phi
        PoissonPhi.setVal(0.);
#ifdef NEUMANN
        // set boundary conditions to homogeneous Neumann
        mlabec.setLevelBC(0, &PoissonPhi);

        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#else
        openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#endif

        // Calculate H from Phi
        ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);
      }
    } else {
      PoissonPhi.setVal(0.);
      PoissonRHS.setVal(0.);
    }

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int plt_step = 0;
	if (restart > 0) {
	  plt_step = restart;
	}
        const std::string& pltfile = amrex::Concatenate("plt",plt_step,8);

        //Averaging face-centerd Multifabs to cell-centers for plotting 
        mf_avg_fc_to_cc(Plt, Mfield, H_biasfield, Ms);
        MultiFab::Copy(Plt, H_demagfield[0], 0, 21, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[1], 0, 22, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[2], 0, 23, 1, 0);
        MultiFab::Copy(Plt, PoissonRHS, 0, 24, 1, 0);
        MultiFab::Copy(Plt, PoissonPhi, 0, 25, 1, 0);

        WriteSingleLevelPlotfile(pltfile, Plt, {"Ms_xface","Ms_yface","Ms_zface",
                                                "Mx_xface","Mx_yface","Mx_zface",
                                                "My_xface", "My_yface", "My_zface",
                                                "Mz_xface", "Mz_yface", "Mz_zface",
                                                "Hx_bias_xface", "Hx_bias_yface", "Hx_bias_zface",
                                                "Hy_bias_xface", "Hy_bias_yface", "Hy_bias_zface",
                                                "Hz_bias_xface", "Hz_bias_yface", "Hz_bias_zface",
                                                "Hx_demagfield","Hy_demagfield","Hz_demagfield",
                                                "PoissonRHS","PoissonPhi"},
                                                 geom, time, plt_step);


    }

    // copy new solution into old solution
    for(int comp = 0; comp < 3; comp++)
    {
       MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 3, Nghost);
       MultiFab::Copy(Mfield_prev_iter[comp], Mfield[comp], 0, 0, 3, Nghost);
       MultiFab::Copy(Mfield_error[comp], Mfield[comp], 0, 0, 3, 0);

       // fill periodic ghost cells
       Mfield_old[comp].FillBoundary(geom.periodicity());
       Mfield_prev_iter[comp].FillBoundary(geom.periodicity());
       Mfield_error[comp].FillBoundary(geom.periodicity());

    }

    TimeIntegrator<Vector<MultiFab> > integrator(Mfield_old);
    
    for (int step = start_step; step <= nsteps; ++step)
    {

        Real step_strt_time = ParallelDescriptor::second();

	// Create a RHS source function we will integrate
        auto source_fun = [&](Vector<MultiFab>& LLG_RHS, const Vector<MultiFab>& Mfield_old, const Real time){
             // User function to calculate the rhs MultiFab given the state MultiFab
    	     // Evolve H_demag
             if(demag_coupling == 1)
             {
                 //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                 //Compute RHS of Poisson equation
                 ComputePoissonRHS(PoissonRHS, Mfield_old, Ms, geom);
                 
                 //Initial guess for phi
                 PoissonPhi.setVal(0.);
#ifdef NEUMANN
                 mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#else
	         openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#endif
    
                 // Calculate H from Phi
                 ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);
             }

             // Compute f^n = f(M^n, H^n) 
             Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, alpha, Ms, gamma, exchange, anisotropy, demag_coupling, exchange_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0, geom, time);
             //fill_rhs(rhs, state, time);
        };

	// Create a function to call after updating a state
        auto post_update_fun = [&](Vector<MultiFab>& Mfield, const Real time) {
             // Call user function to update state MultiFab, e.g. fill BCs
             NormalizeM(Mfield, Ms, M_normalization);

             for(int comp = 0; comp < 3; comp++)
             {
                // fill periodic ghost cells
                Mfield[comp].FillBoundary(geom.periodicity());
             }

            //post_update(Mfield_old, time, geom);
        };

	// Attach the right hand side and post-update functions
        // to the integrator
        integrator.set_rhs(source_fun);
        integrator.set_post_update(post_update_fun);
        
        // integrate forward one step from `time` by `dt` to fill S_new
        integrator.advance(Mfield_old, Mfield, time, dt);

	// copy new solution into old solution
        for(int comp = 0; comp < 3; comp++)
        {
           MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 3, Nghost);

           // fill periodic ghost cells
           Mfield_old[comp].FillBoundary(geom.periodicity());

        }

	Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        // update time
        time = time + dt;

        // Write a plotfile of the data if plot_int > 0
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,8);

            //Averaging face-centerd Multifabs to cell-centers for plotting 
            mf_avg_fc_to_cc(Plt, Mfield, H_biasfield, Ms);
            MultiFab::Copy(Plt, H_demagfield[0], 0, 21, 1, 0);
            MultiFab::Copy(Plt, H_demagfield[1], 0, 22, 1, 0);
            MultiFab::Copy(Plt, H_demagfield[2], 0, 23, 1, 0);
            MultiFab::Copy(Plt, PoissonRHS, 0, 24, 1, 0);
            MultiFab::Copy(Plt, PoissonPhi, 0, 25, 1, 0);

            WriteSingleLevelPlotfile(pltfile, Plt, {"Ms_xface","Ms_yface","Ms_zface",
                                                    "Mx_xface","Mx_yface","Mx_zface",
                                                    "My_xface", "My_yface", "My_zface",
                                                    "Mz_xface", "Mz_yface", "Mz_zface",
                                                    "Hx_bias_xface", "Hx_bias_yface", "Hx_bias_zface",
                                                    "Hy_bias_xface", "Hy_bias_yface", "Hy_bias_zface",
                                                    "Hz_bias_xface", "Hz_bias_yface", "Hz_bias_zface",
                                                    "Hx_demagfield","Hy_demagfield","Hz_demagfield",
                                                    "PoissonRHS","PoissonPhi"},
                                                    geom, time, step);

        }

	// Write a checkpoint file if chk_int > 0
	if (chk_int > 0 && step%chk_int == 0) {
	  WriteCheckPoint(step,time,Mfield,H_biasfield,H_demagfield);
	}

        // MultiFab memory usage
        const int IOProc = ParallelDescriptor::IOProcessorNumber();

        amrex::Long min_fab_megabytes  = amrex::TotalBytesAllocatedInFabsHWM()/1048576;
        amrex::Long max_fab_megabytes  = min_fab_megabytes;

        ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
        ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

        amrex::Print() << "High-water FAB megabyte spread across MPI nodes: ["
                       << min_fab_megabytes << " ... " << max_fab_megabytes << "]\n";

        min_fab_megabytes  = amrex::TotalBytesAllocatedInFabs()/1048576;
        max_fab_megabytes  = min_fab_megabytes;

        ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
        ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

        amrex::Print() << "Curent     FAB megabyte spread across MPI nodes: ["
                       << min_fab_megabytes << " ... " << max_fab_megabytes << "]\n";

    }
    
    Real total_step_stop_time = ParallelDescriptor::second() - total_step_strt_time;
    ParallelDescriptor::ReduceRealMax(total_step_stop_time);

    amrex::Print() << "Total run time " << total_step_stop_time << " seconds\n";
}
