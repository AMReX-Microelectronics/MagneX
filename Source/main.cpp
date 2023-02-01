
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

    Array<MultiFab, AMREX_SPACEDIM> Mfield;
    Array<MultiFab, AMREX_SPACEDIM> H_biasfield;
    Array<MultiFab, AMREX_SPACEDIM> H_demagfield;

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

    Array<MultiFab, AMREX_SPACEDIM> Mfield_old;
    // face-centered Mfield_old
    AMREX_D_TERM(Mfield_old[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
                 Mfield_old[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
                 Mfield_old[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost););

    Array<MultiFab, AMREX_SPACEDIM> Mfield_prev_iter;
    // face-centered Mfield at predictor step (for 2nd order time integrator)
    AMREX_D_TERM(Mfield_prev_iter[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
                 Mfield_prev_iter[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
                 Mfield_prev_iter[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost););

    Array<MultiFab, AMREX_SPACEDIM> Mfield_error;
    // face-centered Mfield at predictor step (for 2nd order time integrator)
    AMREX_D_TERM(Mfield_error[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, 0);,
                 Mfield_error[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, 0);,
                 Mfield_error[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, 0););

    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS;
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

    for (int step = start_step; step <= nsteps; ++step)
    {

        Real step_strt_time = ParallelDescriptor::second();

        if (TimeIntegratorOrder == 1){

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

           //Evolve M
      
           // Compute f^n = f(M^n, H^n) 
           Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, alpha, Ms, gamma, exchange, anisotropy, demag_coupling, exchange_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0, geom);

           // M^{n+1} = M^n + dt * f^n
	   for(int i = 0; i < 3; i++){
	      MultiFab::LinComb(Mfield[i], 1.0, Mfield_old[i], 0, dt, LLG_RHS[i], 0, 0, 3, 0);
	   }

           NormalizeM(Mfield, Ms, M_normalization);

           for(int comp = 0; comp < 3; comp++)
           {
              // fill periodic ghost cells
              Mfield[comp].FillBoundary(geom.periodicity());
           }

           // copy new solution into old solution
           for(int comp = 0; comp < 3; comp++)
           {
              MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 3, Nghost);

              // fill periodic ghost cells
              Mfield_old[comp].FillBoundary(geom.periodicity());

           }
        } else if (TimeIntegratorOrder == 2){ //2nd Order
        amrex::Print() << "TimeIntegratorOrder = " << TimeIntegratorOrder << "\n";

           Real M_tolerance = 1.e-6;
           int iter = 0;
           int stop_iter = 0;

	   // Evolve H_demag (H^{n})
	   if(demag_coupling == 1)
	   {
	      //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
	      //Compute RHS of Poisson equation
	      ComputePoissonRHS(PoissonRHS, Mfield_prev_iter, Ms, geom);
                    
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

	   // Compute f^{n} = f(M^{n}, H^{n}) 
	   Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, alpha, Ms, gamma, exchange, anisotropy, demag_coupling, exchange_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0, geom);

           while(!stop_iter){

  	      // Poisson solve and H_demag computation with M_field_pre
 	      if(demag_coupling == 1)
	      {
		 //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
		 //Compute RHS of Poisson equation
	         ComputePoissonRHS(PoissonRHS, Mfield_prev_iter, Ms, geom);
                    
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

	      // LLG RHS with new H_demag and M_field_pre
	      // Compute f^{n+1, *} = f(M^{n+1, *}, H^{n+1, *}) 
	      Compute_LLG_RHS(LLG_RHS_pre, Mfield_prev_iter, H_demagfield, H_biasfield, alpha, Ms, gamma, exchange, anisotropy, demag_coupling, exchange_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0, geom);

	      // Corrector step update M
	      // M^{n+1, *} = M^n + 0.5 * dt * (f^n + f^{n+1, *})
	      for(int i = 0; i < 3; i++){
		MultiFab::LinComb(LLG_RHS_avg[i], 0.5, LLG_RHS[i], 0, 0.5, LLG_RHS_pre[i], 0, 0, 3, 0);
		MultiFab::LinComb(Mfield[i], 1.0, Mfield_old[i], 0, dt, LLG_RHS_avg[i], 0, 0, 3, 0);
	      }

	      // Normalize M              
	      NormalizeM(Mfield, Ms, M_normalization);

#if 1
              for (MFIter mfi(Ms[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                  Array4<Real> const& Ms_xface = Ms[0].array(mfi);
                  Array4<Real> const& Ms_yface = Ms[1].array(mfi);
                  Array4<Real> const& Ms_zface = Ms[2].array(mfi);
                  
                  Array4<Real> const& Mfield_error_xface = Mfield_error[0].array(mfi);
                  Array4<Real> const& Mfield_error_yface = Mfield_error[1].array(mfi);
                  Array4<Real> const& Mfield_error_zface = Mfield_error[2].array(mfi);

                  Array4<Real> const& Mfield_xface = Mfield[0].array(mfi);
                  Array4<Real> const& Mfield_yface = Mfield[1].array(mfi);
                  Array4<Real> const& Mfield_zface = Mfield[2].array(mfi);

                  Array4<Real> const& Mfield_prev_iter_xface = Mfield_prev_iter[0].array(mfi);
                  Array4<Real> const& Mfield_prev_iter_yface = Mfield_prev_iter[1].array(mfi);
                  Array4<Real> const& Mfield_prev_iter_zface = Mfield_prev_iter[2].array(mfi);

                  Box const &tbx = mfi.tilebox(IntVect(1,0,0));
                  Box const &tby = mfi.tilebox(IntVect(0,1,0));
                  Box const &tbz = mfi.tilebox(IntVect(0,0,1));
                  
                  amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                      if (Ms_xface(i,j,k) > 0) {
                          for (int n=0; n<3; ++n) {
                              Mfield_error_xface(i,j,k,n) = amrex::Math::abs(Mfield_xface(i,j,k,n) - Mfield_prev_iter_xface(i,j,k,n)) / Ms_xface(i,j,k);
                          }
                      } else {
                          for (int n=0; n<3; ++n) {
                              Mfield_error_xface(i,j,k,n) = 0.;
                          }
                      }
                  });
                  
                  amrex::ParallelFor(tby, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                      if (Ms_yface(i,j,k) > 0) {
                          for (int n=0; n<3; ++n) {
                              Mfield_error_yface(i,j,k,n) = amrex::Math::abs(Mfield_yface(i,j,k,n) - Mfield_prev_iter_yface(i,j,k,n)) / Ms_yface(i,j,k);
                          }
                      } else {
                          for (int n=0; n<3; ++n) {
                              Mfield_error_yface(i,j,k,n) = 0.;
                          }
                      }
                  });
                  
                  amrex::ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                      if (Ms_zface(i,j,k) > 0) {
                          for (int n=0; n<3; ++n) {
                              Mfield_error_zface(i,j,k,n) = amrex::Math::abs(Mfield_zface(i,j,k,n) - Mfield_prev_iter_zface(i,j,k,n)) / Ms_zface(i,j,k);
                          }
                      } else {
                          for (int n=0; n<3; ++n) {
                              Mfield_error_zface(i,j,k,n) = 0.;
                          }
                      }
                  });

              }

              amrex::Real M_mag_error_max = -1.;
              for (int face = 0; face < 3; face++){
                  for (int comp = 0; comp < 3; comp++){
                      Real M_iter_error = Mfield_error[face].norm0(comp);
                      if (M_iter_error >= M_mag_error_max){
                          M_mag_error_max = M_iter_error;
                      }
                  }
              }

#else
	      Real M_mag_error_max = -1.;

	      for(int face = 0; face < 3; face++){
 		 for(int comp = 0; comp < 3; comp++){
		    MultiFab::Copy(Mfield_error[face], Mfield[face], 0, 0, 3, 0);
		    MultiFab::Subtract(Mfield_error[face], Mfield_prev_iter[face], 0, 0, 3, 0);
		    Real M_mag_error = Mfield_error[face].norm0(comp)/Mfield[face].norm0(comp);
		    if (M_mag_error >= M_mag_error_max){
		       M_mag_error_max = M_mag_error;
		    }
		 }
	      }
#endif

	      // copy new solution into Mfield_pre_iter
	      for(int comp = 0; comp < 3; comp++)
	      {
	         MultiFab::Copy(Mfield_prev_iter[comp], Mfield[comp], 0, 0, 3, Nghost);
		 // fill periodic ghost cells
		 Mfield_prev_iter[comp].FillBoundary(geom.periodicity());
	      }

	      iter = iter + 1;

	      amrex::Print() << "iter = " << iter << ", M_mag_error_max = " << M_mag_error_max << "\n";
	      if (M_mag_error_max <= M_tolerance) stop_iter = 1;

	   } // while stop_iter

           // copy new solution into old solution
           for (int comp = 0; comp < 3; comp++)
           {
 	      MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 3, Nghost);
	      // fill periodic ghost cells
	      Mfield_old[comp].FillBoundary(geom.periodicity());
           }

        } else if (TimeIntegratorOrder == 3) {

            EvolveM_2nd(Mfield, H_demagfield, H_biasfield, PoissonRHS, PoissonPhi, alpha, Ms, gamma, exchange, anisotropy, demag_coupling, exchange_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0, geom, prob_lo, prob_hi, dt);
        
        }  else {
            amrex::Abort("Time integrator order not recognized");
        }//else 2nd order
 

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
