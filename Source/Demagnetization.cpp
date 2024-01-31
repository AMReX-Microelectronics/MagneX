#include "Demagnetization.H"
#include "MagneX.H"
#include "CartesianAlgorithm_K.H"
#include <AMReX_PlotFileUtil.H>

Demagnetization::Demagnetization() {}

// Compute the Demag tensor in realspace and its FFT
void Demagnetization::define()
{
    // timer for profiling
    BL_PROFILE_VAR("Demagnetization::define()",DemagDefine);

    RealBox real_box_large({AMREX_D_DECL(              prob_lo[0],              prob_lo[1],              prob_lo[2])},
                           {AMREX_D_DECL( 2*prob_hi[0]-prob_lo[0], 2*prob_hi[1]-prob_lo[1], 2*prob_hi[2]-prob_lo[2])});

    // **********************************
    // SIMULATION SETUP
    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    // number of points in the domain, and periodicity

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo_large(AMREX_D_DECL(            0,             0,             0));
    IntVect dom_hi_large(AMREX_D_DECL(2*n_cell[0]-1, 2*n_cell[1]-1, 2*n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain_large(dom_lo_large, dom_hi_large);

    // Initialize the boxarray "ba" from the single box "domain"
    ba_large.define(domain_large);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    // create IntVect of max_grid_size (double the value since this is for the large domain)
    IntVect max_grid_size(AMREX_D_DECL(2*max_grid_size_x,2*max_grid_size_y,2*max_grid_size_z));
    ba_large.maxSize(max_grid_size);

    // How Boxes are distrubuted among MPI processes
    dm_large.define(ba_large);

    // periodic in x and y directions
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)}; // nonperiodic in all directions

    // This defines a Geometry object
    geom_large.define(domain_large, real_box_large, CoordSys::cartesian, is_periodic);

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo_fft(AMREX_D_DECL(            0,             0,             0));
    IntVect dom_hi_fft(AMREX_D_DECL(n_cell[0], 2*n_cell[1]-1, 2*n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain_fft(dom_lo_fft, dom_hi_fft);

    // This defines a Geometry object
    geom_fft.define(domain_fft, real_box_large, CoordSys::cartesian, is_periodic);
    
    if (FFT_solver == 1) {
        // create a BoxArray containing the fft boxes
        // by construction, these boxes correlate to the associated spectral_data
        // this we can copy the spectral data into this multifab since we know they are owned by the same MPI rank
        {
            BoxList bl;
            bl.reserve(ba_large.size());

            for (int i = 0; i < ba_large.size(); ++i) {
                Box b = ba_large[i];

                Box r_box = b;
                Box c_box = amrex::coarsen(r_box, IntVect(AMREX_D_DECL(2,1,1)));

                // this avoids overlap for the cases when one or more r_box's
                // have an even cell index in the hi-x cell
                if (c_box.bigEnd(0) * 2 == r_box.bigEnd(0)) {
                    c_box.setBig(0,c_box.bigEnd(0)-1);
                }

                // increase the size of boxes touching the hi-x domain by 1 in x
                // this is an (Nx x Ny x Nz) -> (Nx/2+1 x Ny x Nz) real-to-complex sizing
                if (b.bigEnd(0) == geom_large.Domain().bigEnd(0)) {
                    c_box.growHi(0,1);
                }
                bl.push_back(c_box);

            }
        ba_fft.define(std::move(bl));
        }

        // Allocate the demag tensor fft multifabs
        Kxx_fft_real.define(ba_fft, dm_large, 1, 0);
        Kxx_fft_imag.define(ba_fft, dm_large, 1, 0);
        Kxy_fft_real.define(ba_fft, dm_large, 1, 0);
        Kxy_fft_imag.define(ba_fft, dm_large, 1, 0);
        Kxz_fft_real.define(ba_fft, dm_large, 1, 0);
        Kxz_fft_imag.define(ba_fft, dm_large, 1, 0);
        Kyy_fft_real.define(ba_fft, dm_large, 1, 0);
        Kyy_fft_imag.define(ba_fft, dm_large, 1, 0);
        Kyz_fft_real.define(ba_fft, dm_large, 1, 0);
        Kyz_fft_imag.define(ba_fft, dm_large, 1, 0);
        Kzz_fft_real.define(ba_fft, dm_large, 1, 0);
        Kzz_fft_imag.define(ba_fft, dm_large, 1, 0);
   
    } else {
	// Allocate the demag tensor fft multifabs
	Kxx_fft_real.define(ba_large, dm_large, 1, 0);
	Kxx_fft_imag.define(ba_large, dm_large, 1, 0);
	Kxy_fft_real.define(ba_large, dm_large, 1, 0);
	Kxy_fft_imag.define(ba_large, dm_large, 1, 0);
	Kxz_fft_real.define(ba_large, dm_large, 1, 0);
	Kxz_fft_imag.define(ba_large, dm_large, 1, 0);
	Kyy_fft_real.define(ba_large, dm_large, 1, 0);
	Kyy_fft_imag.define(ba_large, dm_large, 1, 0);
	Kyz_fft_real.define(ba_large, dm_large, 1, 0);
	Kyz_fft_imag.define(ba_large, dm_large, 1, 0);
	Kzz_fft_real.define(ba_large, dm_large, 1, 0);
	Kzz_fft_imag.define(ba_large, dm_large, 1, 0);
    }

    // Allocate the plot file for the large FFT
    MultiFab Plt (ba_large, dm_large, 6, 0);

    // MultiFab storage for the demag tensor
    // TWICE AS BIG AS THE DOMAIN OF THE PROBLEM!!!!!!!!
    MultiFab Kxx(ba_large, dm_large, 1, 0);
    MultiFab Kxy(ba_large, dm_large, 1, 0);
    MultiFab Kxz(ba_large, dm_large, 1, 0);
    MultiFab Kyy(ba_large, dm_large, 1, 0);
    MultiFab Kyz(ba_large, dm_large, 1, 0);
    MultiFab Kzz(ba_large, dm_large, 1, 0);
    
    Kxx.setVal(0.);
    Kxy.setVal(0.);
    Kxz.setVal(0.);
    Kxz.setVal(0.);
    Kyy.setVal(0.);
    Kyz.setVal(0.);
    Kzz.setVal(0.);

    Real prefactor = 1. / 4. / 3.14159265;

    // Loop through demag tensor and fill with values
    for (MFIter mfi(Kxx); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

	// extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom_large.CellSizeArray();

        const Array4<Real>& Kxx_ptr = Kxx.array(mfi);
        const Array4<Real>& Kxy_ptr = Kxy.array(mfi);
        const Array4<Real>& Kxz_ptr = Kxz.array(mfi);
        const Array4<Real>& Kyy_ptr = Kyy.array(mfi);
        const Array4<Real>& Kyz_ptr = Kyz.array(mfi);
        const Array4<Real>& Kzz_ptr = Kzz.array(mfi);
   
        // Set the demag tensor
	amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int L, int M, int N)
        {   
            // L,M,N range from 0:2*n_cell-1
            // I,J,K range from -n_cell+1:n_cell
            int I = L - n_cell[0] + 1;
            int J = M - n_cell[1] + 1;
            int K = N - n_cell[2] + 1;

            if (I == n_cell[0] || J == n_cell[1] || K == n_cell[2]) {
                return;
            }

            // HACK this cell is coming out differently using integration strategies
            /*
            if (I == 0 && J == 0 && K == 0) {
                return;
            }
            */
            
            // **********************************
            // SET VALUES FOR EACH CELL
            // **********************************
#if 1
            for (int i = 0; i <= 1; i++) { // helper indices
                for (int j = 0; j <= 1; j++) { 
                    for (int k = 0; k <= 1; k++) {
 		        Real r = std::sqrt ((I+i-0.5)*(I+i-0.5)*dx[0]*dx[0] + (J+j-0.5)*(J+j-0.5)*dx[1]*dx[1] + (K+k-0.5)*(K+k-0.5)*dx[2]*dx[2]);
                        
                        Kxx_ptr(L,M,N) = Kxx_ptr(L,M,N) + ((std::pow(-1,i+j+k)) * (std::atan ((K+k-0.5) * (J+j-0.5) * dx[2] * dx[1] / r / (I+i-0.5) / dx[0])));
                        
                        Kxy_ptr(L,M,N) = Kxy_ptr(L,M,N) + ((std::pow(-1,i+j+k)) * (std::log ((K+k-0.5) * dx[2] + r)));
                        
                        Kxz_ptr(L,M,N) = Kxz_ptr(L,M,N) + ((std::pow(-1,i+j+k)) * (std::log ((J+j-0.5) * dx[1] + r)));
                        
                        Kyy_ptr(L,M,N) = Kyy_ptr(L,M,N) + ((std::pow(-1,i+j+k)) * (std::atan ((I+i-0.5) * (K+k-0.5) * dx[0] * dx[2] / r / (J+j-0.5) / dx[1])));
                        
                        Kyz_ptr(L,M,N) = Kyz_ptr(L,M,N) + ((std::pow(-1,i+j+k)) * (std::log ((I+i-0.5) * dx[0] + r)));
                        
                        Kzz_ptr(L,M,N) = Kzz_ptr(L,M,N) + ((std::pow(-1,i+j+k)) * std::atan ((J+j-0.5) * (I+i-0.5) * dx[1] * dx[0] / r / (K+k-0.5) / dx[2]));
                    }
                }
            }
#else
            int sub = 100;
            Real vol = dx[0]*dx[1]*dx[2]/(sub*sub*sub);

            for (int i = -sub/2; i <= sub/2-1; i++) { // helper indices
                for (int j = -sub/2; j <= sub/2-1; j++) {
                    for (int k = -sub/2; k <= sub/2-1; k++) {

                        Real x = I*dx[0] + (i+0.5)*dx[0]/sub;
                        Real y = J*dx[1] + (j+0.5)*dx[1]/sub;
                        Real z = K*dx[2] + (k+0.5)*dx[2]/sub;
                        Real r = std::sqrt(x*x+y*y+z*z);

                        Kxx_ptr(L,M,N) -= (1./(r*r*r)) * (1. - 3.*(x/r)*(x/r)) * vol;
                        Kyy_ptr(L,M,N) -= (1./(r*r*r)) * (1. - 3.*(y/r)*(y/r)) * vol;
                        Kzz_ptr(L,M,N) -= (1./(r*r*r)) * (1. - 3.*(z/r)*(z/r)) * vol;

                        Kxy_ptr(L,M,N) -= (1./(r*r*r)) * (3.*(x/r)*(y/r)) * vol;
                        Kxz_ptr(L,M,N) -= (1./(r*r*r)) * (3.*(x/r)*(z/r)) * vol;
                        Kyz_ptr(L,M,N) -= (1./(r*r*r)) * (3.*(y/r)*(z/r)) * vol;
                    }
                }
            }
#endif
            Kxx_ptr(L,M,N) *= prefactor;
            Kxy_ptr(L,M,N) *= (-prefactor);
            Kxz_ptr(L,M,N) *= (-prefactor);
            Kyy_ptr(L,M,N) *= prefactor;
            Kyz_ptr(L,M,N) *= (-prefactor);
            Kzz_ptr(L,M,N) *= prefactor;

        });
    }

    MultiFab::Copy(Plt, Kxx, 0, 0, 1, 0);
    MultiFab::Copy(Plt, Kxy, 0, 1, 1, 0);
    MultiFab::Copy(Plt, Kxz, 0, 2, 1, 0);
    MultiFab::Copy(Plt, Kyy, 0, 3, 1, 0);
    MultiFab::Copy(Plt, Kyz, 0, 4, 1, 0);
    MultiFab::Copy(Plt, Kzz, 0, 5, 1, 0);

    WriteSingleLevelPlotfile("DemagTensor", Plt,
                             {"Kxx",
                              "Kxy",
                              "Kxz",
                              "Kyy",
                              "Kyz",
                              "Kzz"},
                             geom_large, 0., 0);

    if (FFT_solver == 0) {
        ComputeForwardFFT(Kxx, Kxx_fft_real, Kxx_fft_imag);
        ComputeForwardFFT(Kxy, Kxy_fft_real, Kxy_fft_imag);
        ComputeForwardFFT(Kxz, Kxz_fft_real, Kxz_fft_imag);
        ComputeForwardFFT(Kyy, Kyy_fft_real, Kyy_fft_imag);
        ComputeForwardFFT(Kyz, Kyz_fft_real, Kyz_fft_imag);
        ComputeForwardFFT(Kzz, Kzz_fft_real, Kzz_fft_imag);

    } else {
        ComputeForwardFFT_heffte(Kxx, Kxx_fft_real, Kxx_fft_imag);
        ComputeForwardFFT_heffte(Kxy, Kxy_fft_real, Kxy_fft_imag);
        ComputeForwardFFT_heffte(Kxz, Kxz_fft_real, Kxz_fft_imag);
        ComputeForwardFFT_heffte(Kyy, Kyy_fft_real, Kyy_fft_imag);
        ComputeForwardFFT_heffte(Kyz, Kyz_fft_real, Kyz_fft_imag);
        ComputeForwardFFT_heffte(Kzz, Kzz_fft_real, Kzz_fft_imag);
   }

}

// Convolve the convolution magnetization and the demag tensor by taking the dot product of their FFTs.
// Then take the inverse of that convolution
// Hx = ifftn(fftn(Mx) .* Kxx_fft + fftn(My) .* Kxy_fft + fftn(Mz) .* Kxz_fft); % calc demag field with fft
// Hy = ifftn(fftn(Mx) .* Kxy_fft + fftn(My) .* Kyy_fft + fftn(Mz) .* Kyz_fft);
// Hz = ifftn(fftn(Mx) .* Kxz_fft + fftn(My) .* Kyz_fft + fftn(Mz) .* Kzz_fft);
void Demagnetization::CalculateH_demag(Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                                       Array<MultiFab, AMREX_SPACEDIM>& H_demagfield)
{
    // timer for profiling
    BL_PROFILE_VAR("ComputeH_demag()",ComputeH_demag);

    // copy Mfield into Mfield_padded
    Array<MultiFab, AMREX_SPACEDIM> Mfield_padded;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        Mfield_padded[dir].define(ba_large, dm_large, 1, 0);
        Mfield_padded[dir].setVal(0.);
        Mfield_padded[dir].ParallelCopy(Mfield[dir], 0, 0, 1);
    }

    MultiFab M_dft_real_x;
    MultiFab M_dft_imag_x;
    MultiFab M_dft_real_y;
    MultiFab M_dft_imag_y;
    MultiFab M_dft_real_z;
    MultiFab M_dft_imag_z;
    MultiFab H_dft_real_x;
    MultiFab H_dft_imag_x;
    MultiFab H_dft_real_y;
    MultiFab H_dft_imag_y;
    MultiFab H_dft_real_z;
    MultiFab H_dft_imag_z;
    
    // Calculate the Mx, My, and Mz fft's at the current time step
    // Each fft will be stored in seperate real and imaginary multifabs
    if (FFT_solver == 0) {
      	// Allocate Mfield fft multifabs
	M_dft_real_x.define(ba_large, dm_large, 1, 0);
	M_dft_imag_x.define(ba_large, dm_large, 1, 0);
	M_dft_real_y.define(ba_large, dm_large, 1, 0);
	M_dft_imag_y.define(ba_large, dm_large, 1, 0);
	M_dft_real_z.define(ba_large, dm_large, 1, 0);
	M_dft_imag_z.define(ba_large, dm_large, 1, 0);
	   
        ComputeForwardFFT(Mfield_padded[0], M_dft_real_x, M_dft_imag_x);
        ComputeForwardFFT(Mfield_padded[1], M_dft_real_y, M_dft_imag_y);
        ComputeForwardFFT(Mfield_padded[2], M_dft_real_z, M_dft_imag_z);
	
	// Allocate 6 Multifabs to store the convolutions in Fourier space for H_field
	// This could be done in main but then we have an insane amount of arguments in this function
	H_dft_real_x.define(ba_large, dm_large, 1, 0);
	H_dft_imag_x.define(ba_large, dm_large, 1, 0);
	H_dft_real_y.define(ba_large, dm_large, 1, 0);
	H_dft_imag_y.define(ba_large, dm_large, 1, 0);
	H_dft_real_z.define(ba_large, dm_large, 1, 0);
	H_dft_imag_z.define(ba_large, dm_large, 1, 0);

    } else {
       	M_dft_real_x.define(ba_fft, dm_large, 1, 0);
	M_dft_imag_x.define(ba_fft, dm_large, 1, 0);
	M_dft_real_y.define(ba_fft, dm_large, 1, 0);
	M_dft_imag_y.define(ba_fft, dm_large, 1, 0);
	M_dft_real_z.define(ba_fft, dm_large, 1, 0);
	M_dft_imag_z.define(ba_fft, dm_large, 1, 0);
	
	ComputeForwardFFT_heffte(Mfield_padded[0], M_dft_real_x, M_dft_imag_x);
        ComputeForwardFFT_heffte(Mfield_padded[1], M_dft_real_y, M_dft_imag_y);
        ComputeForwardFFT_heffte(Mfield_padded[2], M_dft_real_z, M_dft_imag_z);
    
    	// Allocate 6 Multifabs to store the convolutions in Fourier space for H_field
	H_dft_real_x.define(ba_fft, dm_large, 1, 0);
	H_dft_imag_x.define(ba_fft, dm_large, 1, 0);
	H_dft_real_y.define(ba_fft, dm_large, 1, 0);
	H_dft_imag_y.define(ba_fft, dm_large, 1, 0);
	H_dft_real_z.define(ba_fft, dm_large, 1, 0);
	H_dft_imag_z.define(ba_fft, dm_large, 1, 0);
    }

    for ( MFIter mfi(Kxx_fft_real); mfi.isValid(); ++mfi )
    {
	const Box& bx = mfi.validbox();
	
	// Declare 6 pointers to the real and imaginary parts of the dft of M in each dimension
	const Array4<Real>& Mx_real = M_dft_real_x.array(mfi);
	const Array4<Real>& Mx_imag = M_dft_imag_x.array(mfi);
	const Array4<Real>& My_real = M_dft_real_y.array(mfi);
	const Array4<Real>& My_imag = M_dft_imag_y.array(mfi);
	const Array4<Real>& Mz_real = M_dft_real_z.array(mfi);
	const Array4<Real>& Mz_imag = M_dft_imag_z.array(mfi);
	
	// Declare 12 pointers to the real and imaginary parts of the dft of K with respect to each partial derivative
	Array4<const Real> Kxx_real = Kxx_fft_real.array(mfi);
	Array4<const Real> Kxx_imag = Kxx_fft_imag.array(mfi);
	Array4<const Real> Kxy_real = Kxy_fft_real.array(mfi);
	Array4<const Real> Kxy_imag = Kxy_fft_imag.array(mfi);
	Array4<const Real> Kxz_real = Kxz_fft_real.array(mfi);
	Array4<const Real> Kxz_imag = Kxz_fft_imag.array(mfi);
	Array4<const Real> Kyy_real = Kyy_fft_real.array(mfi);
	Array4<const Real> Kyy_imag = Kyy_fft_imag.array(mfi);
	Array4<const Real> Kyz_real = Kyz_fft_real.array(mfi);
	Array4<const Real> Kyz_imag = Kyz_fft_imag.array(mfi);
	Array4<const Real> Kzz_real = Kzz_fft_real.array(mfi);
	Array4<const Real> Kzz_imag = Kzz_fft_imag.array(mfi);
	
	// Declare 6 pointers to the real and imaginary parts of the dft of M in each dimension
	const Array4<Real>& H_dft_real_x_ptr = H_dft_real_x.array(mfi);
	const Array4<Real>& H_dft_imag_x_ptr = H_dft_imag_x.array(mfi);
	const Array4<Real>& H_dft_real_y_ptr = H_dft_real_y.array(mfi);
	const Array4<Real>& H_dft_imag_y_ptr = H_dft_imag_y.array(mfi);
	const Array4<Real>& H_dft_real_z_ptr = H_dft_real_z.array(mfi);
	const Array4<Real>& H_dft_imag_z_ptr = H_dft_imag_z.array(mfi);
	
	amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
	{
	    // Take the dot product in fourier space of M and K and store that in 6 different multifabs  
	    H_dft_real_x_ptr(i,j,k) =  Mx_real(i,j,k) * Kxx_real(i,j,k) + My_real(i,j,k) * Kxy_real(i,j,k) + Mz_real(i,j,k) * Kxz_real(i,j,k);
	    H_dft_real_x_ptr(i,j,k) -= Mx_imag(i,j,k) * Kxx_imag(i,j,k) + My_imag(i,j,k) * Kxy_imag(i,j,k) + Mz_imag(i,j,k) * Kxz_imag(i,j,k);

	    H_dft_imag_x_ptr(i,j,k) =  Mx_real(i,j,k) * Kxx_imag(i,j,k) + My_real(i,j,k) * Kxy_imag(i,j,k) + Mz_real(i,j,k) * Kxz_imag(i,j,k);
	    H_dft_imag_x_ptr(i,j,k) += Mx_imag(i,j,k) * Kxx_real(i,j,k) + My_imag(i,j,k) * Kxy_real(i,j,k) + Mz_imag(i,j,k) * Kxz_real(i,j,k);
	    
	    H_dft_real_y_ptr(i,j,k) =  Mx_real(i,j,k) * Kxy_real(i,j,k) + My_real(i,j,k) * Kyy_real(i,j,k) + Mz_real(i,j,k) * Kyz_real(i,j,k);
	    H_dft_real_y_ptr(i,j,k) -= Mx_imag(i,j,k) * Kxy_imag(i,j,k) + My_imag(i,j,k) * Kyy_imag(i,j,k) + Mz_imag(i,j,k) * Kyz_imag(i,j,k);
	    
	    H_dft_imag_y_ptr(i,j,k) =  Mx_real(i,j,k) * Kxy_imag(i,j,k) + My_real(i,j,k) * Kyy_imag(i,j,k) + Mz_real(i,j,k) * Kyz_imag(i,j,k);
	    H_dft_imag_y_ptr(i,j,k) += Mx_imag(i,j,k) * Kxy_real(i,j,k) + My_imag(i,j,k) * Kyy_real(i,j,k) + Mz_imag(i,j,k) * Kyz_real(i,j,k);
	    
	    H_dft_real_z_ptr(i,j,k) =  Mx_real(i,j,k) * Kxz_real(i,j,k) + My_real(i,j,k) * Kyz_real(i,j,k) + Mz_real(i,j,k) * Kzz_real(i,j,k);
	    H_dft_real_z_ptr(i,j,k) -= Mx_imag(i,j,k) * Kxz_imag(i,j,k) + My_imag(i,j,k) * Kyz_imag(i,j,k) + Mz_imag(i,j,k) * Kzz_imag(i,j,k);
	    
	    H_dft_imag_z_ptr(i,j,k) = Mx_real(i,j,k) * Kxz_imag(i,j,k) + My_real(i,j,k) * Kyz_imag(i,j,k) + Mz_real(i,j,k) * Kzz_imag(i,j,k);
	    H_dft_imag_z_ptr(i,j,k) += Mx_imag(i,j,k) * Kxz_real(i,j,k) + My_imag(i,j,k) * Kyz_real(i,j,k) + Mz_imag(i,j,k) * Kzz_real(i,j,k);
	    
	});
     }
    
    MultiFab Hx_large(ba_large, dm_large, 1, 0);
    MultiFab Hy_large(ba_large, dm_large, 1, 0);
    MultiFab Hz_large(ba_large, dm_large, 1, 0);
    
    if (FFT_solver == 0) {
	// Compute the inverse FFT of H_field with respect to the three coordinates and store them in 3 multifabs that this function returns
        ComputeInverseFFT(Hx_large, H_dft_real_x, H_dft_imag_x);
        ComputeInverseFFT(Hy_large, H_dft_real_y, H_dft_imag_y);
        ComputeInverseFFT(Hz_large, H_dft_real_z, H_dft_imag_z);
      
    } else {
	ComputeInverseFFT_heffte(Hx_large, H_dft_real_x, H_dft_imag_x);
        ComputeInverseFFT_heffte(Hy_large, H_dft_real_y, H_dft_imag_y);
        ComputeInverseFFT_heffte(Hz_large, H_dft_real_z, H_dft_imag_z);
    }

    // Copying the elements near the 'upper right' of the double-sized demag back to multifab that is the problem size
    // This is not quite the 'upper right' of the source, it's the destination_coordinate + (n_cell-1)
    MultiBlockIndexMapping dtos;
    dtos.offset = IntVect(1-n_cell[0],1-n_cell[1],1-n_cell[2]); // offset = src - dst; "1-n_cell" because we are shifting downward by n_cell-1
    Box dest_box(IntVect(0),IntVect(n_cell[0]-1,n_cell[1]-1,n_cell[2]-1));
    ParallelCopy(H_demagfield[0], dest_box, Hx_large, 0, 0, 1, IntVect(0), dtos);
    ParallelCopy(H_demagfield[1], dest_box, Hy_large, 0, 0, 1, IntVect(0), dtos);
    ParallelCopy(H_demagfield[2], dest_box, Hz_large, 0, 0, 1, IntVect(0), dtos);
    
}

// FFT for GPUs
// Function accepts a multifab 'mf_in' and computes the FFT, storing it in mf_dft_real amd mf_dft_imag multifabs
void Demagnetization::ComputeForwardFFT_heffte(const MultiFab&    mf_in,
                                               MultiFab&          mf_dft_real,
                                               MultiFab&          mf_dft_imag)
{
    // timer for profiling
    BL_PROFILE_VAR("ComputeForwardFFT_heffte()",ComputeForwardFFT_heffte);

    // **********************************
    // COMPUTE FFT
    // **********************************

    Real time = 0.;
    int step = 0;
    
    // since there is 1 MPI rank per box, here each MPI rank obtains its local box and the associated boxid
    Box local_box;
    int local_boxid;
    {
        for (int i = 0; i < ba_large.size(); ++i) {
            Box b = ba_large[i];
            // each MPI rank has its own local_box Box and local_boxid ID
            if (ParallelDescriptor::MyProc() == dm_large[i]) {
                local_box = b;
                local_boxid = i;
            }
        }
    }

    // now each MPI rank works on its own box
    // for real->complex fft's, the fft is stored in an (nx/2+1) x ny x nz dataset

    // start by coarsening each box by 2 in the x-direction
    Box c_local_box =  amrex::coarsen(local_box, IntVect(AMREX_D_DECL(2,1,1)));

    // if the coarsened box's high-x index is even, we shrink the size in 1 in x
    // this avoids overlap between coarsened boxes
    if (c_local_box.bigEnd(0) * 2 == local_box.bigEnd(0)) {
        c_local_box.setBig(0,c_local_box.bigEnd(0)-1);
    }
    // for any boxes that touch the hi-x domain we
    // increase the size of boxes by 1 in x
    // this makes the overall fft dataset have size (Nx/2+1 x Ny x Nz)
    if (local_box.bigEnd(0) == geom_large.Domain().bigEnd(0)) {
        c_local_box.growHi(0,1);
    }

    // each MPI rank gets storage for its piece of the fft
    BaseFab<GpuComplex<Real> > spectral_field(c_local_box, 1, The_Device_Arena());

#ifdef AMREX_USE_CUDA
    heffte::fft2d_r2c<heffte::backend::cufft> fft
#elif AMREX_USE_HIP
    heffte::fft2d_r2c<heffte::backend::rocfft> fft
#else
    heffte::fft2d_r2c<heffte::backend::fftw> fft
#endif
        ({{local_box.smallEnd(0),local_box.smallEnd(1),local_box.smallEnd(2)},
          {local_box.bigEnd(0)  ,local_box.bigEnd(1)  ,local_box.bigEnd(2)}},
         {{c_local_box.smallEnd(0),c_local_box.smallEnd(1),c_local_box.smallEnd(2)},
          {c_local_box.bigEnd(0)  ,c_local_box.bigEnd(1)  ,c_local_box.bigEnd(2)}},
         0, ParallelDescriptor::Communicator());

    using heffte_complex = typename heffte::fft_output<Real>::type;
    heffte_complex* spectral_data = (heffte_complex*) spectral_field.dataPtr();

    // Perform the FFT and store it in 'spectral_data'
    {
        BL_PROFILE("ForwardTransform_heffte");
        fft.forward(mf_in[local_boxid].dataPtr(), spectral_data);
    }

    // this copies the spectral data into a distributed MultiFab
    for (MFIter mfi(mf_dft_real); mfi.isValid(); ++mfi) {

        Array4<Real> const& realpart = mf_dft_real.array(mfi);
        Array4<Real> const& imagpart = mf_dft_imag.array(mfi);

        Array4< GpuComplex<Real> > spectral = spectral_field.array();

        const Box& bx = mfi.fabbox();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real re = spectral(i,j,k).real();
            Real im = spectral(i,j,k).imag();
	    
	    realpart(i,j,k) = re; 	
	    imagpart(i,j,k) = im;
        });
    }


}

// Serial FFT
// Function accepts a multifab 'mf_in' and computes the FFT, storing it in mf_dft_real amd mf_dft_imag multifabs
void Demagnetization::ComputeForwardFFT(const MultiFab&    mf_in,
                                        MultiFab&          mf_dft_real,
                                        MultiFab&          mf_dft_imag)
{
    // timer for profiling
    BL_PROFILE_VAR("ComputeForwardFFT()",ComputeForwardFFT);

    // **********************************
    // COPY INPUT MULTIFAB INTO A MULTIFAB WITH ONE BOX
    // **********************************

    Box minimalBox = mf_in.boxArray().minimalBox();

    // create a new BoxArray and DistributionMapping for a MultiFab with 1 grid
    BoxArray ba_onegrid(minimalBox);
    DistributionMapping dm_onegrid(ba_onegrid);

    // storage for phi and the dft
    MultiFab mf_in_onegrid      (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_real_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_imag_onegrid(ba_onegrid, dm_onegrid, 1, 0);

    // copy phi into phi_onegrid
    mf_in_onegrid.ParallelCopy(mf_in, 0, 0, 1);

    // **********************************
    // COMPUTE FFT
    // **********************************

#ifdef AMREX_USE_CUDA
    using FFTplan = cufftHandle;
    using FFTcomplex = cuDoubleComplex;
#else
    using FFTplan = fftw_plan;
    using FFTcomplex = fftw_complex;
#endif

    // contain to store FFT - note it is shrunk by "half" in x
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;

    Vector<FFTplan> forward_plan;

    for (MFIter mfi(mf_in_onegrid); mfi.isValid(); ++mfi) {

      // grab a single box including ghost cell range
      Box realspace_bx = mfi.fabbox();

      // size of box including ghost cell range
      IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

      // this is the size of the box, except the 0th component is 'halved plus 1'
      IntVect spectral_bx_size = fft_size;
      spectral_bx_size[0] = fft_size[0]/2 + 1;

      // spectral box
      Box spectral_bx = Box(IntVect(0), spectral_bx_size - IntVect(1));

      spectral_field.emplace_back(new BaseFab<GpuComplex<Real> >(spectral_bx,1,
                                 The_Device_Arena()));
      spectral_field.back()->setVal<RunOn::Device>(0.0); // touch the memory

      FFTplan fplan;

#ifdef AMREX_USE_CUDA

#if (AMREX_SPACEDIM == 2)
      cufftResult result = cufftPlan2d(&fplan, fft_size[1], fft_size[0], CUFFT_D2Z);
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " cufftplan2d forward failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#elif (AMREX_SPACEDIM == 3)
      cufftResult result = cufftPlan3d(&fplan, fft_size[2], fft_size[1], fft_size[0], CUFFT_D2Z);
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " cufftplan3d forward failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#endif

#else // host

#if (AMREX_SPACEDIM == 2)
      fplan = fftw_plan_dft_r2c_2d(fft_size[1], fft_size[0],
                   mf_in_onegrid[mfi].dataPtr(),
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   FFTW_ESTIMATE);
#elif (AMREX_SPACEDIM == 3)
      fplan = fftw_plan_dft_r2c_3d(fft_size[2], fft_size[1], fft_size[0],
                   mf_in_onegrid[mfi].dataPtr(),
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   FFTW_ESTIMATE);
#endif

#endif

      forward_plan.push_back(fplan);
    }

    ParallelDescriptor::Barrier();

    {
        BL_PROFILE("ForwardTransform");
        // ForwardTransform
        for (MFIter mfi(mf_in_onegrid); mfi.isValid(); ++mfi) {
            int i = mfi.LocalIndex();
#ifdef AMREX_USE_CUDA
            cufftSetStream(forward_plan[i], Gpu::gpuStream());
            cufftResult result = cufftExecD2Z(forward_plan[i],
                                              mf_in_onegrid[mfi].dataPtr(),
                                              reinterpret_cast<FFTcomplex*>
                                              (spectral_field[i]->dataPtr()));
            if (result != CUFFT_SUCCESS) {
                AllPrint() << " forward transform using cufftExec failed! Error: "
                           << cufftErrorToString(result) << "\n";
            }
#else
            fftw_execute(forward_plan[i]);
#endif
        }
    }

    // copy data to a full-sized MultiFab
    // this involves copying the complex conjugate from the half-sized field
    // into the appropriate place in the full MultiFab
    for (MFIter mfi(mf_dft_real_onegrid); mfi.isValid(); ++mfi) {

      Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();

      Array4<Real> const& realpart = mf_dft_real_onegrid.array(mfi);
      Array4<Real> const& imagpart = mf_dft_imag_onegrid.array(mfi);

      Box bx = mfi.fabbox();

      amrex:: ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
      /*
        Copying rules:

        For domains from (0,0,0) to (Nx-1,Ny-1,Nz-1)

        For any cells with i index >= Nx/2, these values are complex conjugates of the corresponding
        entry where (Nx-i,Ny-j,Nz-k) UNLESS that index is zero, in which case you use 0.

        e.g. for an 8^3 domain, any cell with i index

        Cell (6,2,3) is complex conjugate of (2,6,5)

        Cell (4,1,0) is complex conjugate of (4,7,0)  (note that the FFT is computed for 0 <= i <= Nx/2)
      */
          if (i <= bx.length(0)/2) {
              // copy value
              realpart(i,j,k) = spectral(i,j,k).real();
              imagpart(i,j,k) = spectral(i,j,k).imag();
          }
          else{
	      // copy complex conjugate
              int iloc = bx.length(0)-i;
              int jloc, kloc;

              jloc = (j == 0) ? 0 : bx.length(1)-j;
#if (AMREX_SPACEDIM == 2)
              kloc = 0;
#elif (AMREX_SPACEDIM == 3)
              kloc = (k == 0) ? 0 : bx.length(2)-k;
#endif

              realpart(i,j,k) =  spectral(iloc,jloc,kloc).real();
              imagpart(i,j,k) = -spectral(iloc,jloc,kloc).imag();
	  }
      });
    }

    // Copy the full multifabs back into the output multifabs
    mf_dft_real.ParallelCopy(mf_dft_real_onegrid, 0, 0, 1);
    mf_dft_imag.ParallelCopy(mf_dft_imag_onegrid, 0, 0, 1);

    // destroy fft plan
    for (int i = 0; i < forward_plan.size(); ++i) {
#ifdef AMREX_USE_CUDA
        cufftDestroy(forward_plan[i]);
#else
        fftw_destroy_plan(forward_plan[i]);
#endif
     }
}

// Inverse FFT for GPUs
// This function takes the real and imaginary parts of data from the frequency domain and performs an inverse FFT, storing the result in 'mf_out'
// The FFTW c2r function is called which accepts complex data in the frequency domain and returns real data in the normal cartesian plane
void Demagnetization::ComputeInverseFFT_heffte(MultiFab&    mf_out,
                                               const MultiFab&          mf_dft_real,
                                               const MultiFab&          mf_dft_imag)
{
    // timer for profiling
    BL_PROFILE_VAR("ComputeInverseFFT_heffte()",ComputeInverseFFT_heffte);

    // **********************************
    // 
    // **********************************

    Box minimalBox = mf_out.boxArray().minimalBox();

    // since there is 1 MPI rank per box, here each MPI rank obtains its local box and the associated boxid
    Box local_box;
    int local_boxid;
    {
        for (int i = 0; i < ba_large.size(); ++i) {
            Box b = ba_large[i];
            // each MPI rank has its own local_box Box and local_boxid ID
            if (ParallelDescriptor::MyProc() == dm_large[i]) {
                local_box = b;
                local_boxid = i;
            }
        }
    }

    // now each MPI rank works on its own box
    // for real->complex fft's, the fft is stored in an (nx/2+1) x ny x nz dataset

    // start by coarsening each box by 2 in the x-direction
    Box c_local_box =  amrex::coarsen(local_box, IntVect(AMREX_D_DECL(2,1,1)));

    // if the coarsened box's high-x index is even, we shrink the size in 1 in x
    // this avoids overlap between coarsened boxes
    if (c_local_box.bigEnd(0) * 2 == local_box.bigEnd(0)) {
        c_local_box.setBig(0,c_local_box.bigEnd(0)-1);
    }
    // for any boxes that touch the hi-x domain we
    // increase the size of boxes by 1 in x
    // this makes the overall fft dataset have size (Nx/2+1 x Ny x Nz)
    if (local_box.bigEnd(0) == geom_large.Domain().bigEnd(0)) {
        c_local_box.growHi(0,1);
    }

    // each MPI rank gets storage for its piece of the fft
    BaseFab<GpuComplex<Real> > spectral_field(c_local_box, 1, The_Device_Arena());

#ifdef AMREX_USE_CUDA
    heffte::fft2d_r2c<heffte::backend::cufft> fft
#elif AMREX_USE_HIP
    heffte::fft2d_r2c<heffte::backend::rocfft> fft
#else
    heffte::fft2d_r2c<heffte::backend::fftw> fft
#endif
        ({{local_box.smallEnd(0),local_box.smallEnd(1),local_box.smallEnd(2)},
          {local_box.bigEnd(0)  ,local_box.bigEnd(1)  ,local_box.bigEnd(2)}},
         {{c_local_box.smallEnd(0),c_local_box.smallEnd(1),c_local_box.smallEnd(2)},
          {c_local_box.bigEnd(0)  ,c_local_box.bigEnd(1)  ,c_local_box.bigEnd(2)}},
         0, ParallelDescriptor::Communicator());

    using heffte_complex = typename heffte::fft_output<Real>::type;
    heffte_complex* spectral_data = (heffte_complex*) spectral_field.dataPtr();

    // this copies the spectral data into a distributed MultiFab
    for (MFIter mfi(mf_dft_real); mfi.isValid(); ++mfi) {

        Array4< const double > realpart = mf_dft_real.array(mfi);
        Array4< const double > imagpart = mf_dft_imag.array(mfi); 
	
	Array4< GpuComplex<Real> > spectral = spectral_field.array();

        const Box& bx = mfi.fabbox();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
	    GpuComplex<Real> copy(realpart(i,j,k),imagpart(i,j,k));
            spectral(i,j,k) = copy;
	    
	});
    }

    {
        BL_PROFILE("BackwardTransform_heffte");
        fft.backward(spectral_data, mf_out[local_boxid].dataPtr());
    }

    // Perform standard scaling after performing FFT
    mf_out.mult(1./(2*n_cell[0] * 2* n_cell[1] * 2*n_cell[2]));

}

// Serial FFT
// This function takes the real and imaginary parts of data from the frequency domain and performs an inverse FFT, storing the result in 'mf_out'
// The FFTW c2r function is called which accepts complex data in the frequency domain and returns real data in the normal cartesian plane
void Demagnetization::ComputeInverseFFT(MultiFab&                        mf_out,
                                        const MultiFab&                  mf_dft_real,
                                        const MultiFab&                  mf_dft_imag)
{
    // timer for profiling
    BL_PROFILE_VAR("ComputeInverseFFT()",ComputeInverseFFT);

    Box minimalBox = mf_out.boxArray().minimalBox();

    // create a new BoxArray and DistributionMapping for a MultiFab with 1 grid
    BoxArray ba_onegrid(minimalBox);
    DistributionMapping dm_onegrid(ba_onegrid);
    
    // Declare multifabs to store entire dataset in one grid.
    MultiFab mf_onegrid_out     (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_real_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_imag_onegrid(ba_onegrid, dm_onegrid, 1, 0);

    // Copy distributed multifabs into one grid multifabs
    mf_dft_real_onegrid.ParallelCopy(mf_dft_real, 0, 0, 1);
    mf_dft_imag_onegrid.ParallelCopy(mf_dft_imag, 0, 0, 1);

#ifdef AMREX_USE_CUDA
    using FFTplan = cufftHandle;
    using FFTcomplex = cuDoubleComplex;
#else
    using FFTplan = fftw_plan;
    using FFTcomplex = fftw_complex;
#endif

    // contain to store FFT - note it is shrunk by "half" in x
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;

    // Copy the contents of the real and imaginary FFT Multifabs into 'spectral_field'
    for (MFIter mfi(mf_dft_real_onegrid); mfi.isValid(); ++mfi) {

      // grab a single box including ghost cell range
      Box realspace_bx = mfi.fabbox();

      // size of box including ghost cell range
      IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

      // this is the size of the box, except the 0th component is 'halved plus 1'
      IntVect spectral_bx_size = fft_size;
      spectral_bx_size[0] = fft_size[0]/2 + 1;

      // spectral box
      Box spectral_bx = Box(IntVect(0), spectral_bx_size - IntVect(1));

      spectral_field.emplace_back(new BaseFab<GpuComplex<Real> >(spectral_bx,1,
                                 The_Device_Arena()));
      spectral_field.back()->setVal<RunOn::Device>(0.0); // touch the memory
      
        // Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();
        Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();

        Array4<Real> const& realpart = mf_dft_real_onegrid.array(mfi);
        Array4<Real> const& imagpart = mf_dft_imag_onegrid.array(mfi);

        Box bx = mfi.fabbox();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {        
            if (i <= bx.length(0)/2) {
                GpuComplex<Real> copy(realpart(i,j,k),imagpart(i,j,k));
                spectral(i,j,k) = copy;
            }   
        });
    }

    // Compute the inverse FFT on spectral_field and store it in 'mf_onegrid_out'
    Vector<FFTplan> backward_plan;

    // Now that we have a spectral field full of the data from the DFT..
    // We perform the inverse DFT on spectral field and store it in mf_onegrid_out
    for (MFIter mfi(mf_onegrid_out); mfi.isValid(); ++mfi) {

       // grab a single box including ghost cell range
       Box realspace_bx = mfi.fabbox();

       // size of box including ghost cell range
       IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

       FFTplan bplan;

#ifdef AMREX_USE_CUDA

#if (AMREX_SPACEDIM == 2)
      cufftResult result = cufftPlan2d(&bplan, fft_size[1], fft_size[0], CUFFT_Z2D);
      if (result != CUFFT_SUCCESS) {
          AllPrint() << " cufftplan2d forward failed! Error: "
                     << cufftErrorToString(result) << "\n";
      }
#elif (AMREX_SPACEDIM == 3)
      cufftResult result = cufftPlan3d(&bplan, fft_size[2], fft_size[1], fft_size[0], CUFFT_Z2D);
      if (result != CUFFT_SUCCESS) {
          AllPrint() << " cufftplan3d forward failed! Error: "
                     << cufftErrorToString(result) << "\n";
      }
#endif

#else // host

#if (AMREX_SPACEDIM == 2)
      bplan = fftw_plan_dft_c2r_2d(fft_size[1], fft_size[0],
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   mf_onegrid_out[mfi].dataPtr(),
                   FFTW_ESTIMATE);
#elif (AMREX_SPACEDIM == 3)
      bplan = fftw_plan_dft_c2r_3d(fft_size[2], fft_size[1], fft_size[0],
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   mf_onegrid_out[mfi].dataPtr(),
                   FFTW_ESTIMATE);
#endif

#endif

      backward_plan.push_back(bplan);// This adds an instance of bplan to the end of backward_plan
    }

    {
        BL_PROFILE("BackwardTransform");
        for (MFIter mfi(mf_onegrid_out); mfi.isValid(); ++mfi) {
            int i = mfi.LocalIndex();
#ifdef AMREX_USE_CUDA
            cufftSetStream(backward_plan[i], Gpu::gpuStream());
            cufftResult result = cufftExecZ2D(backward_plan[i],
                                              reinterpret_cast<FFTcomplex*>
                                              (spectral_field[i]->dataPtr()),
                                              mf_onegrid_out[mfi].dataPtr());
            if (result != CUFFT_SUCCESS) {
                AllPrint() << " inverse transform using cufftExec failed! Error: "
                           << cufftErrorToString(result) << "\n";
            }
#else
            fftw_execute(backward_plan[i]);
#endif
        }
    }

      // Standard scaling after fft and inverse fft using FFTW
#if (AMREX_SPACEDIM == 2)
    mf_onegrid_out.mult(1./(minimalBox.length(0)*minimalBox.length(1)));
#elif (AMREX_SPACEDIM == 3)
    mf_onegrid_out.mult(1./(minimalBox.length(0)*minimalBox.length(1)*minimalBox.length(2)));
#endif

    // copy contents of mf_onegrid_out into mf
    mf_out.ParallelCopy(mf_onegrid_out, 0, 0, 1);

    // destroy ifft plan
    for (int i = 0; i < backward_plan.size(); ++i) {
#ifdef AMREX_USE_CUDA
        cufftDestroy(backward_plan[i]);
#else
        fftw_destroy_plan(backward_plan[i]);
#endif

    }

}

#ifdef AMREX_USE_CUDA
std::string Demagnetization::cufftErrorToString (const cufftResult& err)
{
    switch (err) {
    case CUFFT_SUCCESS:  return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    default: return std::to_string(err) + " (unknown error code)";
    }
}
#endif
