
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 
#include <AMReX_OpenBC.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include "myfunc.H"
#include "MicroMag.H"
#include "MagLaplacian.H"

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


    // **********************************
    // SIMULATION SETUP

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity
    BoxArray ba;
    Geometry geom;

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});

    // periodic in x and y directions
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)}; // nonperiodic in all directions

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // Allocate multifabs

    Array<MultiFab, AMREX_SPACEDIM> Mfield;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        Mfield[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> Mfield_old;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        Mfield_old[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> H_demagfield;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        H_demagfield[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> H_biasfield;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        H_biasfield[dir].define(ba, dm, Ncomp, Nghost);
    }

    MultiFab alpha(ba, dm, Ncomp, Nghost);
    MultiFab gamma(ba, dm, Ncomp, Nghost);
    MultiFab Ms(ba, dm, Ncomp, Nghost);
    MultiFab exchange(ba, dm, Ncomp, Nghost);
    MultiFab anisotropy(ba, dm, Ncomp, Nghost);

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

    MultiFab Plt(ba, dm, 12, 0);

    //Solver for Poisson equation
    LPInfo info;
    OpenBCSolver openbc({geom}, {ba}, {dm}, info);
    openbc.setVerbose(2);
    // openbc.setVerbose(2);

    // time = starting time in the simulation
    Real time = 0.0;	
   
    //Next steps (06/16/2022: Initialze M, solve Poisson's equation for Phi, Compute H from Phi, H_exchane and H_anisotropy from M)

    InitializeMagneticProperties(alpha, Ms, gamma, exchange, anisotropy,
                                 alpha_val, Ms_val, gamma_val, exchange_val, anisotropy_val, 
                                 prob_lo, prob_hi, mag_lo, mag_hi, geom);

    //Initialize fields

    //for (MFIter mfi(*Mfield[0]); mfi.isValid(); ++mfi)
    for (MFIter mfi(Mfield[0]); mfi.isValid(); ++mfi)
    {
    
          const Box& bx = mfi.growntilebox(1); 

    // extract field data
          Array4<Real> const &Mx = Mfield[0].array(mfi);         
          Array4<Real> const &My = Mfield[1].array(mfi);         
          Array4<Real> const &Mz = Mfield[2].array(mfi);
         
          Array4<Real> const &Hx_bias = H_biasfield[0].array(mfi);
          Array4<Real> const &Hy_bias = H_biasfield[1].array(mfi);
          Array4<Real> const &Hz_bias = H_biasfield[2].array(mfi);
      
          Array4<Real> const &Hx_demag= H_demagfield[0].array(mfi);
          Array4<Real> const &Hy_demag= H_demagfield[1].array(mfi);
          Array4<Real> const &Hz_demag= H_demagfield[2].array(mfi);
              
          const Array4<Real>& Ms_arr = Ms.array(mfi);

          amrex::Real angle_theta = 1.5650; // radiant of [111] direction
          amrex::Real angle_phi = 0.2450; // radiant [111] direction

          amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
          {
             if (Ms_arr(i,j,k) > 0._rt)
             {

                Real x = prob_lo[0] + (i+0.5) * dx[0];
                Real y = prob_lo[1] + (j+0.5) * dx[1];
                Real z = prob_lo[2] + (k+0.5) * dx[2];
                
                // Mx(i,j,k) = (y < 0) ? Ms_arr(i,j,k) : 0.;
                // My(i,j,k) = 0._rt;
                // Mz(i,j,k) = (y >= 0) ? Ms_arr(i,j,k) : 0.;

                Mx(i,j,k) = 1/sqrt(3.0)*Ms_arr(i,j,k);
                My(i,j,k) = 1/sqrt(3.0)*Ms_arr(i,j,k);
                Mz(i,j,k) = 1/sqrt(3.0)*Ms_arr(i,j,k);

                // Mx(i,j,k) = 0._rt;
                // My(i,j,k) = Ms_arr(i,j,k);
                // Mz(i,j,k) = 0._rt;

             } else {
                Mx(i,j,k) = 0.0;
                My(i,j,k) = 0.0;
                Mz(i,j,k) = 0.0;
            }
          });

    } 
    if(demag_coupling == 1){ 
        //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
        //Compute RHS of Poisson equation
        ComputePoissonRHS(PoissonRHS, Mfield, Ms, geom);
        
        //Initial guess for phi
        PoissonPhi.setVal(0.);
        openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1); 

        // Calculate H from Phi
        ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);
    }
 
    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,8);
        MultiFab::Copy(Plt, PoissonRHS, 0, 0, 1, 0);  
        MultiFab::Copy(Plt, Ms, 0, 1, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[0], 0, 2, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[1], 0, 3, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[2], 0, 4, 1, 0);
        MultiFab::Copy(Plt, Mfield[0], 0, 5, 1, 0);
        MultiFab::Copy(Plt, Mfield[1], 0, 6, 1, 0);
        MultiFab::Copy(Plt, Mfield[2], 0, 7, 1, 0);
        MultiFab::Copy(Plt, H_biasfield[0], 0, 8, 1, 0);
        MultiFab::Copy(Plt, H_biasfield[1], 0, 9, 1, 0);
        MultiFab::Copy(Plt, H_biasfield[2], 0, 10, 1, 0);
        MultiFab::Copy(Plt, PoissonPhi, 0, 11, 1, 0);
        WriteSingleLevelPlotfile(pltfile, Plt, {"PoissonRHS","Ms","Hx_demag","Hy_demag","Hz_demag","Mx", "My", "Mz", "Hx_bias", "Hy_bias", "Hz_bias", "PoissonPhi"}, geom, time, 0);
    }

    for (int step = 1; step <= nsteps; ++step)
    {
        // copy new solution into old solution
        for(int comp = 0; comp < 3; comp++)
        {
           MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, 1);

           // fill periodic ghost cells
           Mfield_old[comp].FillBoundary(geom.periodicity());

        }

        Real step_strt_time = ParallelDescriptor::second();

    	// Evolve M
        if(demag_coupling == 1)
        {
            //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
            //Compute RHS of Poisson equation
            ComputePoissonRHS(PoissonRHS, Mfield_old, Ms, geom);
            
            //Initial guess for phi
            PoissonPhi.setVal(0.);
            openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
    
            // Calculate H from Phi
            ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);
        }

        //for (MFIter mfi(*Mfield[0]); mfi.isValid(); ++mfi)
        for (MFIter mfi(Mfield[0]); mfi.isValid(); ++mfi)
        {
       
              const Box& bx = mfi.growntilebox(1); 

        // extract field data
              Array4<Real> const &Hx_demag= H_demagfield[0].array(mfi);
              Array4<Real> const &Hy_demag= H_demagfield[1].array(mfi);
              Array4<Real> const &Hz_demag= H_demagfield[2].array(mfi);
              Array4<Real> const &Mx = Mfield[0].array(mfi);         
              Array4<Real> const &My = Mfield[1].array(mfi);         
              Array4<Real> const &Mz = Mfield[2].array(mfi);         
              Array4<Real> const &Mx_old = Mfield_old[0].array(mfi); 
              Array4<Real> const &My_old = Mfield_old[1].array(mfi); 
              Array4<Real> const &Mz_old = Mfield_old[2].array(mfi); 
              Array4<Real> const &Hx_bias = H_biasfield[0].array(mfi);
              Array4<Real> const &Hy_bias = H_biasfield[1].array(mfi);
              Array4<Real> const &Hz_bias = H_biasfield[2].array(mfi);
          
              const Array4<Real>& alpha_arr = alpha.array(mfi);
              const Array4<Real>& gamma_arr = gamma.array(mfi);
              const Array4<Real>& Ms_arr = Ms.array(mfi);
              const Array4<Real>& exchange_arr = exchange.array(mfi);
              const Array4<Real>& anisotropy_arr = anisotropy.array(mfi);

              amrex::Real t0 = 1.0e-9; // time when bias reduces to zero
              amrex::Real frequency = 2.e9;
              amrex::Real TP = 1/frequency;
              amrex::Real pi = 3.14159265358979;
 
              amrex::ParallelForRNG( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
              {
                 if (Ms_arr(i,j,k) > 0._rt)
                 {

                    if (time <= t0){
                        Hx_bias(i,j,k) = 1.0e5 * (1-time/t0) * 1._rt/sqrt(3);
                        Hy_bias(i,j,k) = 1.0e5 * (1-time/t0) * 1._rt/sqrt(3);
                        Hz_bias(i,j,k) = 1.0e5 * (1-time/t0) * 1._rt/sqrt(3);
                    } else {
                        Hx_bias(i,j,k) = 0.0 ;
                        Hy_bias(i,j,k) = 0.0 ;
                        Hz_bias(i,j,k) = 0.0 ;
                    }

                    // amrex::Real z = prob_lo[2] + (k+0.5) * dx[2];
                    
                    // Hx_bias(i,j,k) = 24.0 * (exp(-(time-3.* TP)*(time-3.* TP)/(2*TP*TP))*cos(2*pi*frequency*time)) * cos(z / 345.0e-9 * pi);
                    // Hy_bias(i,j,k) = 2.4e4;
                    // Hz_bias(i,j,k) = 0.;

                    // Hx_bias(i,j,k) = (j < 32) ? 1.0e5 : 0.;
                    // Hy_bias(i,j,k) = (j < 32) ? 0. : 1.0e5;
                    // Hz_bias(i,j,k) = 0.;

                    amrex::Real Hx_eff = Hx_bias(i,j,k);
                    amrex::Real Hy_eff = Hy_bias(i,j,k);
                    amrex::Real Hz_eff = Hz_bias(i,j,k);

                    Hx_eff += (-1.0 + 2.0*Random(engine))*0.002; // add random noise
                    Hy_eff += (-1.0 + 2.0*Random(engine))*0.002; // add random noise
                    Hz_eff += (-1.0 + 2.0*Random(engine))*0.002; // add random noise
                    
                    if(demag_coupling == 1)
                    {
                        Hx_eff += Hx_demag(i,j,k);
                        Hy_eff += Hy_demag(i,j,k);
                        Hz_eff += Hz_demag(i,j,k);

                        if (i == 128 && j == 32 && k == 4){
                            printf("got here \n");
                            printf("Hx_demag = %g", Hx_demag(i,j,k));
                            printf("Hy_demag = %g", Hy_demag(i,j,k));
                            printf("Hz_demag = %g", Hz_demag(i,j,k));
                        }
                        
                    }

                    if(exchange_coupling == 1)
                    { 
                    //Add exchange term
                      if (exchange_arr(i,j,k) == 0._rt) amrex::Abort("The exchange_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                      // H_exchange - use M^(old_time)
                      amrex::Real const H_exchange_coeff = 2.0 * exchange_arr(i,j,k) / mu0 / Ms_arr(i,j,k) / Ms_arr(i,j,k);

                      amrex::Real Ms_lo_x = Ms_arr(i-1, j, k); 
                      amrex::Real Ms_hi_x = Ms_arr(i+1, j, k); 
                      amrex::Real Ms_lo_y = Ms_arr(i, j-1, k); 
                      amrex::Real Ms_hi_y = Ms_arr(i, j+1, k); 
                      amrex::Real Ms_lo_z = Ms_arr(i, j, k-1);
                      amrex::Real Ms_hi_z = Ms_arr(i, j, k+1);

                      if(i == 31 && j == 31 && k == 31) printf("Laplacian_x = %g \n", Laplacian_Mag(Mx_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx));
                      Hx_eff += H_exchange_coeff * Laplacian_Mag(Mx_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx);
                      Hy_eff += H_exchange_coeff * Laplacian_Mag(My_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx);
                      Hz_eff += H_exchange_coeff * Laplacian_Mag(Mz_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx);

                    }
                 
                    if(anisotropy_coupling == 1)
                    {
                     //Add anisotropy term
 
                     if (anisotropy_arr(i,j,k) == 0._rt) amrex::Abort("The anisotropy_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                      // H_anisotropy - use M^(old_time)
                      amrex::Real M_dot_anisotropy_axis = 0.0;
                      M_dot_anisotropy_axis = Mx(i, j, k) * anisotropy_axis[0] + My(i, j, k) * anisotropy_axis[1] + Mz(i, j, k) * anisotropy_axis[2];
                      amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_arr(i,j,k) / mu0 / Ms_arr(i,j,k) / Ms_arr(i,j,k);
                      Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                      Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                      Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];

                    }

                   //Update M

                   amrex::Real mag_gammaL = gamma_arr(i,j,k) / (1._rt + std::pow(alpha_arr(i,j,k), 2._rt));

                   // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s 
                   amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(Mx(i, j, k), 2._rt) + std::pow(My(i, j, k), 2._rt) + std::pow(Mz(i, j, k), 2._rt))
                                                             : Ms_arr(i,j,k);
                   amrex::Real Gil_damp = mu0 * mag_gammaL * alpha_arr(i,j,k) / M_magnitude;

                   // x component on cell-centers
                   Mx(i, j, k) += dt * (mu0 * mag_gammaL) * (My_old(i, j, k) * Hz_eff - Mz_old(i, j, k) * Hy_eff)
                                        + dt * Gil_damp * (My_old(i, j, k) * (Mx_old(i, j, k) * Hy_eff - My_old(i, j, k) * Hx_eff)
                                        - Mz_old(i, j, k) * (Mz_old(i, j, k) * Hx_eff - Mx_old(i, j, k) * Hz_eff));

                   // y component on cell-centers
                   My(i, j, k) += dt * (mu0 * mag_gammaL) * (Mz_old(i, j, k) * Hx_eff - Mx_old(i, j, k) * Hz_eff)
                                        + dt * Gil_damp * (Mz_old(i, j, k) * (My_old(i, j, k) * Hz_eff - Mz_old(i, j, k) * Hy_eff)
                                        - Mx_old(i, j, k) * (Mx_old(i, j, k) * Hy_eff - My_old(i, j, k) * Hx_eff));

                   // z component on cell-centers
                   Mz(i, j, k) += dt * (mu0 * mag_gammaL) * (Mx_old(i, j, k) * Hy_eff - My_old(i, j, k) * Hx_eff)
                                        + dt * Gil_damp * (Mx_old(i, j, k) * (Mz_old(i, j, k) * Hx_eff - Mx_old(i, j, k) * Hz_eff)
                                        - My_old(i, j, k) * (My_old(i, j, k) * Hz_eff - Mz_old(i, j, k) * Hy_eff));
  

                   // temporary normalized magnitude of M_xface field at the fixed point
                   amrex::Real M_magnitude_normalized = std::sqrt(std::pow(Mx(i, j, k), 2._rt) + std::pow(My(i, j, k), 2._rt) + std::pow(Mz(i, j, k), 2._rt)) / Ms_arr(i,j,k);

                       
                   amrex::Real normalized_error = 0.1;

                   if (M_normalization > 0)
                   {
                       // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                       // check the normalized error
                       if (amrex::Math::abs(1._rt - M_magnitude_normalized) > normalized_error)
                       {
                           printf("M_magnitude_normalized = %g \n", M_magnitude_normalized);
                           printf("i = %d, j = %d, k = %d \n", i, j,k);
                           amrex::Abort("Exceed the normalized error of the Mx field");
                       }
                       // normalize the M field
                       Mx(i, j, k) /= M_magnitude_normalized;
                       My(i, j, k) /= M_magnitude_normalized;
                       Mz(i, j, k) /= M_magnitude_normalized;
                   }
                   else if (M_normalization == 0)
                   {   
		       if(i == 1 && j == 1 && k == 1) printf("Here ??? \n");
                       // check the normalized error
                       if (M_magnitude_normalized > (1._rt + normalized_error))
                       {
                           amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                       }
                       else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= (1._rt + normalized_error) )
                       {
                           // normalize the M field
                           Mx(i, j, k) /= M_magnitude_normalized;
                           My(i, j, k) /= M_magnitude_normalized;
                           Mz(i, j, k) /= M_magnitude_normalized;
                       }
                   }

                 }   
 
              });     
                      
     }  

	Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        // update time
        time = time + dt;

        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,8);
            MultiFab::Copy(Plt, PoissonRHS, 0, 0, 1, 0);  
            MultiFab::Copy(Plt, Ms, 0, 1, 1, 0);
            MultiFab::Copy(Plt, H_demagfield[0], 0, 2, 1, 0);
            MultiFab::Copy(Plt, H_demagfield[1], 0, 3, 1, 0);
            MultiFab::Copy(Plt, H_demagfield[2], 0, 4, 1, 0);
            MultiFab::Copy(Plt, Mfield[0], 0, 5, 1, 0);
            MultiFab::Copy(Plt, Mfield[1], 0, 6, 1, 0);
            MultiFab::Copy(Plt, Mfield[2], 0, 7, 1, 0);
            MultiFab::Copy(Plt, H_biasfield[0], 0, 8, 1, 0);
            MultiFab::Copy(Plt, H_biasfield[1], 0, 9, 1, 0);
            MultiFab::Copy(Plt, H_biasfield[2], 0, 10, 1, 0);
            MultiFab::Copy(Plt, PoissonPhi, 0, 11, 1, 0);
            WriteSingleLevelPlotfile(pltfile, Plt, {"PoissonRHS","Ms","Hx_demag","Hy_demag","Hz_demag","Mx", "My", "Mz", "Hx_bias", "Hy_bias", "Hz_bias", "PoissonPhi"}, geom, time, step);
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
