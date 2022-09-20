
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
    // face-centered Mfield
    AMREX_D_TERM(Mfield[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
                 Mfield[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
                 Mfield[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost););

    Array<MultiFab, AMREX_SPACEDIM> Mfield_old;
    // face-centered Mfield_old
    AMREX_D_TERM(Mfield_old[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
                 Mfield_old[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
                 Mfield_old[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost););

    Array<MultiFab, AMREX_SPACEDIM> Mfield_cc;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        Mfield_cc[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> H_demagfield;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        H_demagfield[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> H_biasfield;
    // face-centered H_biasfield
    AMREX_D_TERM(H_biasfield[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
                 H_biasfield[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
                 H_biasfield[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost););

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
    AMREX_D_TERM(Ms[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, Nghost);,
                 Ms[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, Nghost);,
                 Ms[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, Nghost););

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

    MultiFab Plt(ba, dm, 21, 0);

    //Solver for Poisson equation
    LPInfo info;
    OpenBCSolver openbc({geom}, {ba}, {dm}, info);
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
        const Array4<Real>& M_xface = Mfield[0].array(mfi);         
        const Array4<Real>& M_yface = Mfield[1].array(mfi);         
        const Array4<Real>& M_zface = Mfield[2].array(mfi);
         
        const Array4<Real>& H_bias_xface = H_biasfield[0].array(mfi);
        const Array4<Real>& H_bias_yface = H_biasfield[1].array(mfi);
        const Array4<Real>& H_bias_zface = H_biasfield[2].array(mfi);
      
        const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
        const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
        const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

        // extract tileboxes for which to loop
        Box const &tbx = mfi.tilebox(Mfield[0].ixType().toIntVect());
        Box const &tby = mfi.tilebox(Mfield[1].ixType().toIntVect());
        Box const &tbz = mfi.tilebox(Mfield[2].ixType().toIntVect());

        amrex::ParallelFor( tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

            if (Ms_xface_arr(i,j,k) > 0._rt)
            {

                Real x = prob_lo[0] + i * dx[0];
                Real y = prob_lo[1] + (j+0.5) * dx[1];
                Real z = prob_lo[2] + (k+0.5) * dx[2];
               
                //x_face 
                M_xface(i,j,k,0) = (y < 0) ? 1.4e5 : 0.;
                M_xface(i,j,k,1) = 0._rt;
                M_xface(i,j,k,2) = (y >= 0) ? 1.4e5 : 0.;

                H_bias_xface(i,j,k,0) = 0._rt;         
                H_bias_xface(i,j,k,1) = 3.7e4;
                H_bias_xface(i,j,k,2) = 0._rt;

             } else {
             
                //x_face 
                M_xface(i,j,k,0) = 0.0; 
                M_xface(i,j,k,1) = 0.0;
                M_xface(i,j,k,2) = 0.0;

                H_bias_xface(i,j,k,0) = 0.0;         
                H_bias_xface(i,j,k,1) = 0.0;
                H_bias_xface(i,j,k,2) = 0.0;

	     }
 

        });

        amrex::ParallelFor( tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

            if (Ms_yface_arr(i,j,k) > 0._rt)
            {

                Real x = prob_lo[0] + (i+0.5) * dx[0];
                Real y = prob_lo[1] + j * dx[1];
                Real z = prob_lo[2] + (k+0.5) * dx[2];
               
                //y_face
                M_yface(i,j,k,0) = (y < 0) ? 1.4e5 : 0.;
                M_yface(i,j,k,1) = 0._rt;
                M_yface(i,j,k,2) = (y >= 0) ? 1.4e5 : 0.;

                H_bias_yface(i,j,k,0) = 0._rt;         
                H_bias_yface(i,j,k,1) = 3.7e4;
                H_bias_yface(i,j,k,2) = 0._rt;

             } else {
             
                //y_face
                M_yface(i,j,k,0) = 0.0;
                M_yface(i,j,k,1) = 0.0;
                M_yface(i,j,k,2) = 0.0;

                H_bias_yface(i,j,k,0) = 0.0;         
                H_bias_yface(i,j,k,1) = 0.0;
                H_bias_yface(i,j,k,2) = 0.0;

	     }

        });

        amrex::ParallelFor( tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

            if (Ms_zface_arr(i,j,k) > 0._rt)
            {

                Real x = prob_lo[0] + (i+0.5) * dx[0];
                Real y = prob_lo[1] + (j+0.5) * dx[1];
                Real z = prob_lo[2] + k * dx[2];
               
                //z_face
                M_zface(i,j,k,0) = (y < 0) ? 1.4e5 : 0.;
                M_zface(i,j,k,1) = 0._rt;
                M_zface(i,j,k,2) = (y >= 0) ? 1.4e5 : 0.;

                H_bias_zface(i,j,k,0) = 0._rt;         
                H_bias_zface(i,j,k,1) = 3.7e4;
                H_bias_zface(i,j,k,2) = 0._rt;

             } else {
             
                //z_face
                M_zface(i,j,k,0) = 0.0;
                M_zface(i,j,k,1) = 0.0;
                M_zface(i,j,k,2) = 0.0;

                H_bias_zface(i,j,k,0) = 0.0;         
                H_bias_zface(i,j,k,1) = 0.0;
                H_bias_zface(i,j,k,2) = 0.0;

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
        //Averaging face-centerd Multifabs to cell-centers for plotting 
        for (MFIter mfi(Plt); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox(); 

            // extract field data
            const Array4<Real>& M_xface = Mfield[0].array(mfi);         
            const Array4<Real>& M_yface = Mfield[1].array(mfi);         
            const Array4<Real>& M_zface = Mfield[2].array(mfi);
             
            const Array4<Real>& H_bias_xface = H_biasfield[0].array(mfi);
            const Array4<Real>& H_bias_yface = H_biasfield[1].array(mfi);
            const Array4<Real>& H_bias_zface = H_biasfield[2].array(mfi);
          
            const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
            const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
            const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

            const Array4<Real>& plt = Plt.array(mfi);

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                //Ms at xface, yface, zface
                plt(i,j,k,0) = 0.5 * ( Ms_xface_arr(i,j,k) + Ms_xface_arr(i+1,j,k) );   
                plt(i,j,k,1) = 0.5 * ( Ms_yface_arr(i,j,k) + Ms_yface_arr(i,j+1,k) );   
                plt(i,j,k,2) = 0.5 * ( Ms_zface_arr(i,j,k) + Ms_zface_arr(i,j,k+1) );   

                //Mx at xface, yface, zface
                plt(i,j,k,3) = 0.5 * ( M_xface(i,j,k,0) + M_xface(i+1,j,k,0) );   
                plt(i,j,k,4) = 0.5 * ( M_yface(i,j,k,0) + M_yface(i,j+1,k,0) );   
                plt(i,j,k,5) = 0.5 * ( M_zface(i,j,k,0) + M_zface(i,j,k+1,0) );  
 
                //My at xface, yface, zface
                plt(i,j,k,6) = 0.5 * ( M_xface(i,j,k,1) + M_xface(i+1,j,k,1) );   
                plt(i,j,k,7) = 0.5 * ( M_yface(i,j,k,1) + M_yface(i,j+1,k,1) );   
                plt(i,j,k,8) = 0.5 * ( M_zface(i,j,k,1) + M_zface(i,j,k+1,1) );  
 
                //Mz at xface, yface, zface
                plt(i,j,k,9)  = 0.5 * ( M_xface(i,j,k,2) + M_xface(i+1,j,k,2) );   
                plt(i,j,k,10) = 0.5 * ( M_yface(i,j,k,2) + M_yface(i,j+1,k,2) );   
                plt(i,j,k,11) = 0.5 * ( M_zface(i,j,k,2) + M_zface(i,j,k+1,2) );  
 
                //Hx_bias at xface, yface, zface
                plt(i,j,k,12) = 0.5 * ( H_bias_xface(i,j,k,0) + H_bias_xface(i+1,j,k,0) );   
                plt(i,j,k,13) = 0.5 * ( H_bias_yface(i,j,k,0) + H_bias_yface(i,j+1,k,0) );   
                plt(i,j,k,14) = 0.5 * ( H_bias_zface(i,j,k,0) + H_bias_zface(i,j,k+1,0) );  
 
                //Hy_bias at xface, yface, zface
                plt(i,j,k,15) = 0.5 * ( H_bias_xface(i,j,k,1) + H_bias_xface(i+1,j,k,1) );   
                plt(i,j,k,16) = 0.5 * ( H_bias_yface(i,j,k,1) + H_bias_yface(i,j+1,k,1) );   
                plt(i,j,k,17) = 0.5 * ( H_bias_zface(i,j,k,1) + H_bias_zface(i,j,k+1,1) );  
 
                //Hz_bias at xface, yface, zface
                plt(i,j,k,18) = 0.5 * ( H_bias_xface(i,j,k,2) + H_bias_xface(i+1,j,k,2) );   
                plt(i,j,k,19) = 0.5 * ( H_bias_yface(i,j,k,2) + H_bias_yface(i,j+1,k,2) );   
                plt(i,j,k,20) = 0.5 * ( H_bias_zface(i,j,k,2) + H_bias_zface(i,j,k+1,2) );  
 
            });

        } 
        WriteSingleLevelPlotfile(pltfile, Plt, {"Ms_xface","Ms_yface","Ms_zface",
                                                "Mx_xface","Mx_yface","Mx_zface",
                                                "My_xface", "My_yface", "My_zface",
                                                "Mz_xface", "Mz_yface", "Mz_zface",
                                                "Hx_bias_xface", "Hx_bias_yface", "Hx_bias_zface",
                                                "Hy_bias_xface", "Hy_bias_yface", "Hy_bias_zface",
                                                "Hz_bias_xface", "Hz_bias_yface", "Hz_bias_zface"},
                                                 geom, time, step);

    }

    for (int step = 1; step <= nsteps; ++step)
    {
        // copy new solution into old solution
        for(int comp = 0; comp < 3; comp++)
        {
           MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 3, 1);

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
            const Array4<Real>& Hx_demag= H_demagfield[0].array(mfi);
            const Array4<Real>& Hy_demag= H_demagfield[1].array(mfi);
            const Array4<Real>& Hz_demag= H_demagfield[2].array(mfi);

            const Array4<Real>& M_xface = Mfield[0].array(mfi);         
            const Array4<Real>& M_yface = Mfield[1].array(mfi);         
            const Array4<Real>& M_zface = Mfield[2].array(mfi);         

            const Array4<Real>& M_xface_old = Mfield_old[0].array(mfi); 
            const Array4<Real>& M_yface_old = Mfield_old[1].array(mfi); 
            const Array4<Real>& M_zface_old = Mfield_old[2].array(mfi); 

            const Array4<Real>& H_bias_xface = H_biasfield[0].array(mfi);
            const Array4<Real>& H_bias_yface = H_biasfield[1].array(mfi);
            const Array4<Real>& H_bias_zface = H_biasfield[2].array(mfi);
          
            const Array4<Real>& alpha_xface_arr = alpha[0].array(mfi);
            const Array4<Real>& alpha_yface_arr = alpha[1].array(mfi);
            const Array4<Real>& alpha_zface_arr = alpha[2].array(mfi);

            const Array4<Real>& gamma_xface_arr = gamma[0].array(mfi);
            const Array4<Real>& gamma_yface_arr = gamma[1].array(mfi);
            const Array4<Real>& gamma_zface_arr = gamma[2].array(mfi);

            const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
            const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
            const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

            const Array4<Real>& exchange_xface_arr = exchange[0].array(mfi);
            const Array4<Real>& exchange_yface_arr = exchange[1].array(mfi);
            const Array4<Real>& exchange_zface_arr = exchange[2].array(mfi);

            const Array4<Real>& anisotropy_xface_arr = anisotropy[0].array(mfi);
            const Array4<Real>& anisotropy_yface_arr = anisotropy[1].array(mfi);
            const Array4<Real>& anisotropy_zface_arr = anisotropy[2].array(mfi);

            amrex::IntVect Mxface_stag = Mfield[0].ixType().toIntVect();
            amrex::IntVect Myface_stag = Mfield[1].ixType().toIntVect();
            amrex::IntVect Mzface_stag = Mfield[2].ixType().toIntVect();

            // extract tileboxes for which to loop
            Box const &tbx = mfi.tilebox(Mfield[0].ixType().toIntVect());
            Box const &tby = mfi.tilebox(Mfield[1].ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Mfield[2].ixType().toIntVect());

            //xface 
            amrex::ParallelFor( tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_xface_arr(i,j,k) > 0._rt)
                {
                    amrex::Real Hx_eff = H_bias_xface(i,j,k,0);
                    amrex::Real Hy_eff = H_bias_xface(i,j,k,1);
                    amrex::Real Hz_eff = H_bias_xface(i,j,k,2);

                    // PSSW validation
                    // amrex::Real z = prob_lo[2] + (k+0.5) * dx[2];
                    
                    // Hx_bias(i,j,k) = 24.0 * (exp(-(time-3.* TP)*(time-3.* TP)/(2*TP*TP))*cos(2*pi*frequency*time)) * cos(z / 345.0e-9 * pi);
                    // Hy_bias(i,j,k) = 2.4e4;
                    // Hz_bias(i,j,k) = 0.;

                    if(demag_coupling == 1)
                    {
                        Hx_eff += face_avg_to_face(i, j, k, 0, Mxface_stag, Mxface_stag, Hx_demag);
                        Hy_eff += face_avg_to_face(i, j, k, 0, Myface_stag, Mxface_stag, Hy_demag);
                        Hz_eff += face_avg_to_face(i, j, k, 0, Mzface_stag, Mxface_stag, Hz_demag);
                    }

                    if(exchange_coupling == 1)
                    { 
                    //Add exchange term
                      if (exchange_xface_arr(i,j,k) == 0._rt) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                      // H_exchange - use M^(old_time)
                      amrex::Real const H_exchange_coeff = 2.0 * exchange_xface_arr(i,j,k) / mu0 / Ms_xface_arr(i,j,k) / Ms_xface_arr(i,j,k);

                      amrex::Real Ms_lo_x = Ms_xface_arr(i-1, j, k); 
                      amrex::Real Ms_hi_x = Ms_xface_arr(i+1, j, k); 
                      amrex::Real Ms_lo_y = Ms_xface_arr(i, j-1, k); 
                      amrex::Real Ms_hi_y = Ms_xface_arr(i, j+1, k); 
                      amrex::Real Ms_lo_z = Ms_xface_arr(i, j, k-1);
                      amrex::Real Ms_hi_z = Ms_xface_arr(i, j, k+1);

		      //if(i == 31 && j == 31 && k == 31) printf("Laplacian_x = %g \n", Laplacian_Mag(M_xface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 0));
                      Hx_eff += H_exchange_coeff * Laplacian_Mag(M_xface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 0);
                      Hy_eff += H_exchange_coeff * Laplacian_Mag(M_xface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 1, 0);
                      Hz_eff += H_exchange_coeff * Laplacian_Mag(M_xface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 2, 0);

                    }
                 
                    if(anisotropy_coupling == 1)
                    {
                     //Add anisotropy term
 
                     if (anisotropy_xface_arr(i,j,k) == 0._rt) amrex::Abort("The anisotropy_xface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                      // H_anisotropy - use M^(old_time)
                      amrex::Real M_dot_anisotropy_axis = 0.0;
                      for (int comp=0; comp<3; ++comp) {
                          M_dot_anisotropy_axis += M_xface_old(i, j, k, comp) * anisotropy_axis[comp];
                      }
                      amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_xface_arr(i,j,k) / mu0 / Ms_xface_arr(i,j,k) / Ms_xface_arr(i,j,k);
                      Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                      Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                      Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];

                    }

                   //Update M

                   amrex::Real mag_gammaL = gamma_xface_arr(i,j,k) / (1._rt + std::pow(alpha_xface_arr(i,j,k), 2._rt));

                   // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                   amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt))
                                                              : Ms_xface_arr(i,j,k); 
                   amrex::Real Gil_damp = mu0 * mag_gammaL * alpha_xface_arr(i,j,k) / M_magnitude;

                   // x component on x-faces of grid
                    M_xface(i, j, k, 0) += dt * (mu0 * mag_gammaL) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 1) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff)
                                         - M_xface_old(i, j, k, 2) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff));

                    // y component on x-faces of grid
                    M_xface(i, j, k, 1) += dt * (mu0 * mag_gammaL) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 2) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff)
                                         - M_xface_old(i, j, k, 0) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff));

                    // z component on x-faces of grid
                    M_xface(i, j, k, 2) += dt * (mu0 * mag_gammaL) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 0) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff)
                                         - M_xface_old(i, j, k, 1) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff));

                   // temporary normalized magnitude of M_xface field at the fixed point
                   amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) / Ms_xface_arr(i,j,k);
                   amrex::Real normalized_error = 0.1;

                   if (M_normalization > 0)
                   {
                       // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                       // check the normalized error
                       if (amrex::Math::abs(1._rt - M_magnitude_normalized) > normalized_error)
                       {
                           printf("M_magnitude_normalized = %g \n", M_magnitude_normalized);
                           amrex::Abort("Exceed the normalized error of the Mx field");
                       }
                       // normalize the M field
                       M_xface(i, j, k, 0) /= M_magnitude_normalized;
                       M_xface(i, j, k, 1) /= M_magnitude_normalized;
                       M_xface(i, j, k, 2) /= M_magnitude_normalized;
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
                           M_xface(i, j, k, 0) /= M_magnitude_normalized;
                           M_xface(i, j, k, 1) /= M_magnitude_normalized;
                           M_xface(i, j, k, 2) /= M_magnitude_normalized;
                       }
                   }

                 }   
 
            });     
                      
            //yface 
            amrex::ParallelFor( tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_yface_arr(i,j,k) > 0._rt)
                {
                    amrex::Real Hx_eff = H_bias_yface(i,j,k,0);
                    amrex::Real Hy_eff = H_bias_yface(i,j,k,1);
                    amrex::Real Hz_eff = H_bias_yface(i,j,k,2);
                 
                    if(demag_coupling == 1)
                    {
                      Hx_eff += face_avg_to_face(i, j, k, 0, Mxface_stag, Myface_stag, Hx_demag);
                      Hy_eff += face_avg_to_face(i, j, k, 0, Myface_stag, Myface_stag, Hy_demag);
                      Hz_eff += face_avg_to_face(i, j, k, 0, Mzface_stag, Myface_stag, Hz_demag);
                    }

                    if(exchange_coupling == 1)
                    { 
                    //Add exchange term
                      if (exchange_yface_arr(i,j,k) == 0._rt) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                      // H_exchange - use M^(old_time)
                      amrex::Real const H_exchange_coeff = 2.0 * exchange_yface_arr(i,j,k) / mu0 / Ms_yface_arr(i,j,k) / Ms_yface_arr(i,j,k);

                      amrex::Real Ms_lo_x = Ms_yface_arr(i-1, j, k); 
                      amrex::Real Ms_hi_x = Ms_yface_arr(i+1, j, k); 
                      amrex::Real Ms_lo_y = Ms_yface_arr(i, j-1, k); 
                      amrex::Real Ms_hi_y = Ms_yface_arr(i, j+1, k); 
                      amrex::Real Ms_lo_z = Ms_yface_arr(i, j, k-1);
                      amrex::Real Ms_hi_z = Ms_yface_arr(i, j, k+1);

		      //if(i == 31 && j == 31 && k == 31) printf("Laplacian_x = %g \n", Laplacian_Mag(M_yface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 0));
                      Hx_eff += H_exchange_coeff * Laplacian_Mag(M_yface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 1);
                      Hy_eff += H_exchange_coeff * Laplacian_Mag(M_yface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 1, 1);
                      Hz_eff += H_exchange_coeff * Laplacian_Mag(M_yface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 2, 1);

                    }
                 
                    if(anisotropy_coupling == 1)
                    {
                     //Add anisotropy term
 
                     if (anisotropy_yface_arr(i,j,k) == 0._rt) amrex::Abort("The anisotropy_yface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                      // H_anisotropy - use M^(old_time)
                      amrex::Real M_dot_anisotropy_axis = 0.0;
                      for (int comp=0; comp<3; ++comp) {
                          M_dot_anisotropy_axis += M_yface_old(i, j, k, comp) * anisotropy_axis[comp];
                      }
                      amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_yface_arr(i,j,k) / mu0 / Ms_yface_arr(i,j,k) / Ms_yface_arr(i,j,k);
                      Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                      Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                      Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];

                    }

                   //Update M

                   amrex::Real mag_gammaL = gamma_yface_arr(i,j,k) / (1._rt + std::pow(alpha_yface_arr(i,j,k), 2._rt));

                   // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                   amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt))
                                                              : Ms_yface_arr(i,j,k); 
                   amrex::Real Gil_damp = mu0 * mag_gammaL * alpha_yface_arr(i,j,k) / M_magnitude;

                   // x component on y-faces of grid
                    M_yface(i, j, k, 0) += dt * (mu0 * mag_gammaL) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 1) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff)
                                         - M_yface_old(i, j, k, 2) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff));

                    // y component on y-faces of grid
                    M_yface(i, j, k, 1) += dt * (mu0 * mag_gammaL) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 2) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff)
                                         - M_yface_old(i, j, k, 0) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff));

                    // z component on y-faces of grid
                    M_yface(i, j, k, 2) += dt * (mu0 * mag_gammaL) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 0) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff)
                                         - M_yface_old(i, j, k, 1) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff));

                   // temporary normalized magnitude of M_xface field at the fixed point
                   amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) / Ms_yface_arr(i,j,k);
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
                       M_yface(i, j, k, 0) /= M_magnitude_normalized;
                       M_yface(i, j, k, 1) /= M_magnitude_normalized;
                       M_yface(i, j, k, 2) /= M_magnitude_normalized;
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
                           M_yface(i, j, k, 0) /= M_magnitude_normalized;
                           M_yface(i, j, k, 1) /= M_magnitude_normalized;
                           M_yface(i, j, k, 2) /= M_magnitude_normalized;
                       }
                   }

                 }   
 
            });     
                      
            //zface 
            amrex::ParallelFor( tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_zface_arr(i,j,k) > 0._rt)
                {
                    amrex::Real Hx_eff = H_bias_zface(i,j,k,0);
                    amrex::Real Hy_eff = H_bias_zface(i,j,k,1);
                    amrex::Real Hz_eff = H_bias_zface(i,j,k,2);
                 
                    if(demag_coupling == 1)
                    {
                      Hx_eff += face_avg_to_face(i, j, k, 0, Mxface_stag, Mzface_stag, Hx_demag);
                      Hy_eff += face_avg_to_face(i, j, k, 0, Myface_stag, Mzface_stag, Hy_demag);
                      Hz_eff += face_avg_to_face(i, j, k, 0, Mzface_stag, Mzface_stag, Hz_demag);
                    }

                    if(exchange_coupling == 1)
                    { 
                    //Add exchange term
                      if (exchange_zface_arr(i,j,k) == 0._rt) amrex::Abort("The exchange_zface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                      // H_exchange - use M^(old_time)
                      amrex::Real const H_exchange_coeff = 2.0 * exchange_zface_arr(i,j,k) / mu0 / Ms_zface_arr(i,j,k) / Ms_zface_arr(i,j,k);

                      amrex::Real Ms_lo_x = Ms_zface_arr(i-1, j, k); 
                      amrex::Real Ms_hi_x = Ms_zface_arr(i+1, j, k); 
                      amrex::Real Ms_lo_y = Ms_zface_arr(i, j-1, k); 
                      amrex::Real Ms_hi_y = Ms_zface_arr(i, j+1, k); 
                      amrex::Real Ms_lo_z = Ms_zface_arr(i, j, k-1);
                      amrex::Real Ms_hi_z = Ms_zface_arr(i, j, k+1);

		      //if(i == 31 && j == 31 && k == 31) printf("Laplacian_x = %g \n", Laplacian_Mag(M_zface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 0));
                      Hx_eff += H_exchange_coeff * Laplacian_Mag(M_zface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 2);
                      Hy_eff += H_exchange_coeff * Laplacian_Mag(M_zface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 1, 2);
                      Hz_eff += H_exchange_coeff * Laplacian_Mag(M_zface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 2, 2);

                    }
                 
                    if(anisotropy_coupling == 1)
                    {
                     //Add anisotropy term
 
                     if (anisotropy_zface_arr(i,j,k) == 0._rt) amrex::Abort("The anisotropy_zface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                      // H_anisotropy - use M^(old_time)
                      amrex::Real M_dot_anisotropy_axis = 0.0;
                      for (int comp=0; comp<3; ++comp) {
                          M_dot_anisotropy_axis += M_zface_old(i, j, k, comp) * anisotropy_axis[comp];
                      }
                      amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_zface_arr(i,j,k) / mu0 / Ms_zface_arr(i,j,k) / Ms_zface_arr(i,j,k);
                      Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                      Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                      Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];

                    }

                   //Update M

                   amrex::Real mag_gammaL = gamma_zface_arr(i,j,k) / (1._rt + std::pow(alpha_zface_arr(i,j,k), 2._rt));

                   // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                   amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt))
                                                              : Ms_zface_arr(i,j,k); 
                   amrex::Real Gil_damp = mu0 * mag_gammaL * alpha_zface_arr(i,j,k) / M_magnitude;

                   // x component on z-faces of grid
                    M_zface(i, j, k, 0) += dt * (mu0 * mag_gammaL) * (M_zface_old(i, j, k, 1) * Hz_eff - M_zface_old(i, j, k, 2) * Hy_eff)
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 1) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff)
                                         - M_zface_old(i, j, k, 2) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff));

                    // y component on z-faces of grid
                    M_zface(i, j, k, 1) += dt * (mu0 * mag_gammaL) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff)
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 2) * (M_zface_old(i, j, k, 1) * Hz_eff - M_zface_old(i, j, k, 2) * Hy_eff)
                                         - M_zface_old(i, j, k, 0) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff));

                    // z component on z-faces of grid
                    M_zface(i, j, k, 2) += dt * (mu0 * mag_gammaL) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff)
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 0) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff)
                                         - M_zface_old(i, j, k, 1) * (M_zface_old(i, j, k, 1) * Hz_eff - M_zface_old(i, j, k, 2) * Hy_eff));

                   // temporary normalized magnitude of M_xface field at the fixed point
                   amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) / Ms_zface_arr(i,j,k);
                   amrex::Real normalized_error = 0.1;

                   if (M_normalization > 0)
                   {
                       // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                       // check the normalized error
                       if (amrex::Math::abs(1._rt - M_magnitude_normalized) > normalized_error)
                       {
                           printf("M_magnitude_normalized = %g \n", M_magnitude_normalized);
                           amrex::Abort("Exceed the normalized error of the Mx field");
                       }
                       // normalize the M field
                       M_zface(i, j, k, 0) /= M_magnitude_normalized;
                       M_zface(i, j, k, 1) /= M_magnitude_normalized;
                       M_zface(i, j, k, 2) /= M_magnitude_normalized;
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
                           M_zface(i, j, k, 0) /= M_magnitude_normalized;
                           M_zface(i, j, k, 1) /= M_magnitude_normalized;
                           M_zface(i, j, k, 2) /= M_magnitude_normalized;
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

        // Write a plotfile of the initial data if plot_int > 0
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,8);
            //Averaging face-centerd Multifabs to cell-centers for plotting 
            for (MFIter mfi(Plt); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox(); 

                // extract field data
                const Array4<Real>& M_xface = Mfield[0].array(mfi);         
                const Array4<Real>& M_yface = Mfield[1].array(mfi);         
                const Array4<Real>& M_zface = Mfield[2].array(mfi);
                 
                const Array4<Real>& H_bias_xface = H_biasfield[0].array(mfi);
                const Array4<Real>& H_bias_yface = H_biasfield[1].array(mfi);
                const Array4<Real>& H_bias_zface = H_biasfield[2].array(mfi);
              
                const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
                const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
                const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

                const Array4<Real>& plt = Plt.array(mfi);

                amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    //Ms at xface, yface, zface
                    plt(i,j,k,0) = 0.5 * ( Ms_xface_arr(i,j,k) + Ms_xface_arr(i+1,j,k) );   
                    plt(i,j,k,1) = 0.5 * ( Ms_yface_arr(i,j,k) + Ms_yface_arr(i,j+1,k) );   
                    plt(i,j,k,2) = 0.5 * ( Ms_zface_arr(i,j,k) + Ms_zface_arr(i,j,k+1) );   
 
                    //Mx at xface, yface, zface
                    plt(i,j,k,3) = 0.5 * ( M_xface(i,j,k,0) + M_xface(i+1,j,k,0) );   
                    plt(i,j,k,4) = 0.5 * ( M_yface(i,j,k,0) + M_yface(i,j+1,k,0) );   
                    plt(i,j,k,5) = 0.5 * ( M_zface(i,j,k,0) + M_zface(i,j,k+1,0) );  
 
                    //My at xface, yface, zface
                    plt(i,j,k,6) = 0.5 * ( M_xface(i,j,k,1) + M_xface(i+1,j,k,1) );   
                    plt(i,j,k,7) = 0.5 * ( M_yface(i,j,k,1) + M_yface(i,j+1,k,1) );   
                    plt(i,j,k,8) = 0.5 * ( M_zface(i,j,k,1) + M_zface(i,j,k+1,1) );  
 
                    //Mz at xface, yface, zface
                    plt(i,j,k,9)  = 0.5 * ( M_xface(i,j,k,2) + M_xface(i+1,j,k,2) );   
                    plt(i,j,k,10) = 0.5 * ( M_yface(i,j,k,2) + M_yface(i,j+1,k,2) );   
                    plt(i,j,k,11) = 0.5 * ( M_zface(i,j,k,2) + M_zface(i,j,k+1,2) );  
 
                    //Hx_bias at xface, yface, zface
                    plt(i,j,k,12) = 0.5 * ( H_bias_xface(i,j,k,0) + H_bias_xface(i+1,j,k,0) );   
                    plt(i,j,k,13) = 0.5 * ( H_bias_yface(i,j,k,0) + H_bias_yface(i,j+1,k,0) );   
                    plt(i,j,k,14) = 0.5 * ( H_bias_zface(i,j,k,0) + H_bias_zface(i,j,k+1,0) );  
 
                    //Hy_bias at xface, yface, zface
                    plt(i,j,k,15) = 0.5 * ( H_bias_xface(i,j,k,1) + H_bias_xface(i+1,j,k,1) );   
                    plt(i,j,k,16) = 0.5 * ( H_bias_yface(i,j,k,1) + H_bias_yface(i,j+1,k,1) );   
                    plt(i,j,k,17) = 0.5 * ( H_bias_zface(i,j,k,1) + H_bias_zface(i,j,k+1,1) );  
 
                    //Hz_bias at xface, yface, zface
                    plt(i,j,k,18) = 0.5 * ( H_bias_xface(i,j,k,2) + H_bias_xface(i+1,j,k,2) );   
                    plt(i,j,k,19) = 0.5 * ( H_bias_yface(i,j,k,2) + H_bias_yface(i,j+1,k,2) );   
                    plt(i,j,k,20) = 0.5 * ( H_bias_zface(i,j,k,2) + H_bias_zface(i,j,k+1,2) );  
 
                });

            } 
            WriteSingleLevelPlotfile(pltfile, Plt, {"Ms_xface","Ms_yface","Ms_zface",
                                                    "Mx_xface","Mx_yface","Mx_zface",
                                                    "My_xface", "My_yface", "My_zface",
                                                    "Mz_xface", "Mz_yface", "Mz_zface",
                                                    "Hx_bias_xface", "Hx_bias_yface", "Hx_bias_zface",
                                                    "Hy_bias_xface", "Hy_bias_yface", "Hy_bias_zface",
                                                    "Hz_bias_xface", "Hz_bias_yface", "Hz_bias_zface"},
                                                     geom, time, step);

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
