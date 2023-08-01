#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif

#include "MagnetostaticSolver.H"
#include "CartesianAlgorithm.H"

void ComputePoissonRHS(MultiFab&                        PoissonRHS,
                       Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                       MultiFab& Ms,
                       const Geometry&                 geom)
{
    for ( MFIter mfi(PoissonRHS); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real const>& Mx = Mfield[0].array(mfi);         
            const Array4<Real const>& My = Mfield[1].array(mfi);         
            const Array4<Real const>& Mz = Mfield[2].array(mfi);   

            const Array4<Real>& rhs = PoissonRHS.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

               rhs(i,j,k) =  DivergenceDx_Mag(Mx, i, j, k, dx)
                           + DivergenceDy_Mag(My, i, j, k, dx)
                           + DivergenceDz_Mag(Mz, i, j, k, dx);
                
            });
        }

}

void ComputeHfromPhi(MultiFab&                        PoissonPhi,
                     Array<MultiFab, AMREX_SPACEDIM>& H_demagfield,
                     amrex::GpuArray<amrex::Real, 3>  prob_lo,
                     amrex::GpuArray<amrex::Real, 3>  prob_hi,
                     const Geometry&                  geom)
{
       // Calculate H from Phi

        for ( MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real>& Hx_demag = H_demagfield[0].array(mfi);
            const Array4<Real>& Hy_demag = H_demagfield[1].array(mfi);
            const Array4<Real>& Hz_demag = H_demagfield[2].array(mfi);

            const Array4<Real>& phi = PoissonPhi.array(mfi);

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                     Hx_demag(i,j,k) = -(phi(i+1,j,k) - phi(i-1,j,k))/2.0/(dx[0]);
                     Hy_demag(i,j,k) = -(phi(i,j+1,k) - phi(i,j-1,k))/2.0/(dx[1]);
                     Hz_demag(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k-1))/2.0/(dx[2]); // consider using GetGradSolution function from amrex
             });
        }

}




// Function accepts the geometry of the problem and then defines the demagnetization tensor in space.  
void ComputeDemagTensor(MultiFab&                        Kxx,
		        MultiFab&                        Kxy,
		        MultiFab&                        Kxz,
		        MultiFab&                        Kyy,
		        MultiFab&                        Kyz,
		        MultiFab&                        Kzz,
			GpuArray<int, 3>                 n_cell,
                        const Geometry&                  geom)
{
    Real prefactor = 1. / 4. / 3.14159265;

    Kxx.setVal(0.);
    Kxy.setVal(0.);
    Kxz.setVal(0.);
    Kxz.setVal(0.);
    Kyy.setVal(0.);
    Kyz.setVal(0.);
    Kzz.setVal(0.);

    // Real r = 0.;

    // Loop through demag tensor and fill with values
    for (MFIter mfi(Kxx); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

	// extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real>& Kxx_ptr = Kxx.array(mfi);
        const Array4<Real>& Kxy_ptr = Kxy.array(mfi);
        const Array4<Real>& Kxz_ptr = Kxz.array(mfi);
        const Array4<Real>& Kyy_ptr = Kyy.array(mfi);
        const Array4<Real>& Kyz_ptr = Kyz.array(mfi);
        const Array4<Real>& Kzz_ptr = Kzz.array(mfi);
   
        // Set the demag tensor
	amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int L, int M, int N)
        {   
	    /*
            if (L == n_cell[0] && M == n_cell[1] && N == n_cell[2]){
                continue;
            }

	    if (L == 0 || M == 0 || N == 0){
                continue;
            }

	    if (L || 2*n_cell[0] || M == 2*n_cell[1] || N == 2*n_cell[2]){
                continue;
            }
	    */

            // Need a negative notion of index where demag is centered at the origin, so we make an aritificial copy of it
            int I = L - n_cell[0];
            int J = M - n_cell[1];
            int K = N - n_cell[2];

            // **********************************
            // SET VALUES FOR EACH CELL
            // **********************************
            for (int i = 0; i <= 1; i++) { // helper indices
                for (int j = 0; j <= 1; j++) { 
                    for (int k = 0; k <= 1; k++) { 
                        Real r = std::sqrt ((I+i-0.5)*(I+i-0.5)*dx[0]*dx[0] + (J+j-0.5)*(J+j-0.5)*dx[1]*dx[1] + (K+k-0.5)*(K+k-0.5)*dx[2]*dx[2]);
                        
                        Kxx_ptr(L,M,N) = Kxx_ptr(L,M,N) + std::pow(-1,i+j+k) * std::atan ((K+k-0.5) * (J+j-0.5) * dx[2] * dx[1] / r / (I+i-0.5) / dx[0]);
                        
                        Kxy_ptr(L,M,N) = Kxy_ptr(L,M,N) + std::pow(-1,i+j+k) * std::log ((K+k-0.5) * dx[2] + r);
                        
                        Kxz_ptr(L,M,N) = Kxz_ptr(L,M,N) + std::pow(-1,i+j+k) * std::log ((J+j-0.5) * dx[1] + r);
                        
                        Kyy_ptr(L,M,N) = Kyy_ptr(L,M,N) + std::pow(-1,i+j+k) * std::atan ((I+i-0.5) * (K+k-0.5) * dx[0] * dx[2] / r / (J+j-0.5) / dx[1]);
                        
                        Kyz_ptr(L,M,N) = Kyz_ptr(L,M,N) + std::pow(-1,i+j+k) * std::log ((I+i-0.5) * dx[0] + r);
                        
                        Kzz_ptr(L,M,N) = Kzz_ptr(L,M,N) + std::pow(-1,i+j+k) * std::atan ((J+j-0.5) * (I+i-0.5) * dx[1] * dx[0] / r / (K+k-0.5) / dx[2]);
                    }
                }
            }

            Kxx_ptr(L,M,N) *= prefactor;
            Kxy_ptr(L,M,N) *= (-prefactor);
            Kxz_ptr(L,M,N) *= (-prefactor);
            Kyy_ptr(L,M,N) *= prefactor;
            Kyz_ptr(L,M,N) *= (-prefactor);
            Kzz_ptr(L,M,N) *= prefactor;

        });
    }

    
}

// THIS COMES LAST!!!!!!!!! COULD BE THE TRICKY PART...
// We will call the three other functions from within this function... which will be called from 'main.cpp' at each time step
// First we take fft's of M_field...
// Then take convolution of the fft of magnetization and the fft of demag tensor 
// Then take the inverse of that convolution
// These steps have the form outlined below 
// Hx = ifftn(fftn(Mx) .* Kxx_fft + fftn(My) .* Kxy_fft + fftn(Mz) .* Kxz_fft); % calc demag field with fft
// Hy = ifftn(fftn(Mx) .* Kxy_fft + fftn(My) .* Kyy_fft + fftn(Mz) .* Kyz_fft);
// Hz = ifftn(fftn(Mx) .* Kxz_fft + fftn(My) .* Kyz_fft + fftn(Mz) .* Kzz_fft);
void ComputeHFieldFFT(const Array<MultiFab, AMREX_SPACEDIM>& M_field,
	              Array<MultiFab, AMREX_SPACEDIM>&       H_demagfield,
                      const MultiFab&                        Kxx_fft_real,
		      const MultiFab&                        Kxx_fft_imag,
                      const MultiFab&                        Kxy_fft_real,
		      const MultiFab&                        Kxy_fft_imag,
 		      const MultiFab&                        Kxz_fft_real,
		      const MultiFab&                        Kxz_fft_imag,
		      const MultiFab&                        Kyy_fft_real,
		      const MultiFab&                        Kyy_fft_imag,
		      const MultiFab&                        Kyz_fft_real,
		      const MultiFab&                        Kyz_fft_imag,
		      const MultiFab&                        Kzz_fft_real,
		      const MultiFab&                        Kzz_fft_imag,
                      GpuArray<Real, 3>                      prob_lo,
                      GpuArray<Real, 3>                      prob_hi,
                      GpuArray<int, 3>                       n_cell,
		      int                                    max_grid_size,
                      const Geometry&                        geom)
{

    BoxArray ba;

    // define lower and upper indices
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    long npts = domain.numPts();

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // This defines the physical box size in each direction
    RealBox real_box({ AMREX_D_DECL(prob_lo[0], prob_lo[1], prob_lo[2])},
                     { AMREX_D_DECL(prob_hi[0], prob_hi[1], prob_hi[2])} );

    // Allocate M_field fft multifabs
    MultiFab M_dft_real_x(ba, dm, 1, 0);
    MultiFab M_dft_imag_x(ba, dm, 1, 0);
    MultiFab M_dft_real_y(ba, dm, 1, 0);
    MultiFab M_dft_imag_y(ba, dm, 1, 0);
    MultiFab M_dft_real_z(ba, dm, 1, 0);
    MultiFab M_dft_imag_z(ba, dm, 1, 0);

    // Calculate the Mx, My, and Mz fft's at the current time step
    // Each fft will be stored in seperate real and imaginary multifabs
    ComputeForwardFFT(M_field[0], M_dft_real_x, M_dft_imag_x, geom, npts);
    ComputeForwardFFT(M_field[1], M_dft_real_y, M_dft_imag_y, geom, npts);
    ComputeForwardFFT(M_field[2], M_dft_real_z, M_dft_imag_z, geom, npts);

    // Allocate 6 Multifabs to store the convolutions in Fourier space for H_field
    // This could be done in main but then we have an insane amount of arguments in this function
    MultiFab H_dft_real_x(ba, dm, 1, 0);
    MultiFab H_dft_imag_x(ba, dm, 1, 0);
    MultiFab H_dft_real_y(ba, dm, 1, 0);
    MultiFab H_dft_imag_y(ba, dm, 1, 0);
    MultiFab H_dft_real_z(ba, dm, 1, 0);
    MultiFab H_dft_imag_z(ba, dm, 1, 0);

    for ( MFIter mfi(M_field[0]); mfi.isValid(); ++mfi )
    {
            const Box& bx = mfi.validbox();

	    // Declare 6 pointers to the real and imaginary parts of the dft of M in each dimension
            const Array4<Real>& Mx_real = M_dft_real_x.array(mfi);
	    const Array4<Real>& Mx_imag = M_dft_imag_x.array(mfi);
            const Array4<Real>& My_real = M_dft_real_y.array(mfi);
	    const Array4<Real>& My_imag = M_dft_imag_y.array(mfi);
            const Array4<Real>& Mz_real = M_dft_real_z.array(mfi);
	    const Array4<Real>& Mz_imag = M_dft_imag_z.array(mfi);

            // Declare 6 pointers to the real and imaginary parts of the dft of M in each dimension
            const Array4<Real>& H_dft_real_x_ptr = H_dft_real_x.array(mfi);
            const Array4<Real>& H_dft_imag_x_ptr = H_dft_imag_x.array(mfi);
            const Array4<Real>& H_dft_real_y_ptr = H_dft_real_y.array(mfi);
            const Array4<Real>& H_dft_imag_y_ptr = H_dft_imag_y.array(mfi);
            const Array4<Real>& H_dft_real_z_ptr = H_dft_real_z.array(mfi);
            const Array4<Real>& H_dft_imag_z_ptr = H_dft_imag_z.array(mfi);


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

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Take the dot product in fourier space of M and K and store that in 6 different multifabs  
	        H_dft_real_x_ptr(i,j,k) = Mx_real(i,j,k) * Kxx_real(i,j,k) + My_real(i,j,k) * Kxy_real(i,j,k) + Mz_real(i,j,k) * Kxz_real(i,j,k);
	        H_dft_imag_x_ptr(i,j,k) = Mx_real(i,j,k) * Kxx_imag(i,j,k) + My_imag(i,j,k) * Kxy_imag(i,j,k) + Mz_imag(i,j,k) * Kxz_imag(i,j,k);
                
		H_dft_real_y_ptr(i,j,k) = Mx_real(i,j,k) * Kxy_real(i,j,k) + My_real(i,j,k) * Kyy_real(i,j,k) + Mz_real(i,j,k) * Kyz_real(i,j,k);
                H_dft_imag_y_ptr(i,j,k) = Mx_real(i,j,k) * Kxy_imag(i,j,k) + My_imag(i,j,k) * Kyy_imag(i,j,k) + Mz_imag(i,j,k) * Kyz_imag(i,j,k);
                
		H_dft_real_z_ptr(i,j,k) = Mx_real(i,j,k) * Kxz_real(i,j,k) + My_real(i,j,k) * Kyz_real(i,j,k) + Mz_real(i,j,k) * Kzz_real(i,j,k);
                H_dft_imag_z_ptr(i,j,k) = Mx_real(i,j,k) * Kxz_imag(i,j,k) + My_imag(i,j,k) * Kyz_imag(i,j,k) + Mz_imag(i,j,k) * Kzz_imag(i,j,k);
         
	    });
     }

    // Allocate Multifabs to store double-sized H_field
    // MultiFab Hx(ba, dm, 1, 0);
    // MultiFab Hy(ba, dm, 1, 0);
    // MultiFab Hz(ba, dm, 1, 0);

    // Compute the inverse FFT of H_field with respect to the three coordinates and store them in 3 multifabs that this function returns
    ComputeInverseFFT(H_demagfield[0], H_dft_real_x, H_dft_imag_x, n_cell, geom);
    ComputeInverseFFT(H_demagfield[1], H_dft_real_y, H_dft_imag_y, n_cell, geom);
    ComputeInverseFFT(H_demagfield[2], H_dft_real_z, H_dft_imag_z, n_cell, geom); 


    /*
    // Copying the elements of the double_sized demag back to multifab that is the problem size
    for ( MFIter mfi(M_field[0]); mfi.isValid(); ++mfi )
    {
            const Box& bx = mfi.validbox();

            // Declare 6 pointers to the real and imaginary parts of the dft of M in each dimension
            const Array4<Real>& Hx_large = Hx.array(mfi);
            const Array4<Real>& Hy_large = Hy.array(mfi);
            const Array4<Real>& Hz_large = Hz.array(mfi);

	    const Array4<Real>& Hx_small = H_demagfield[0].array(mfi);
            const Array4<Real>& Hy_small = H_demagfield[1].array(mfi);
            const Array4<Real>& Hz_small = H_demagfield[2].array(mfi);

  
  	    amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (i >= n_cell[0] && j >= n_cell[1] && k >= n_cell[2]){
		    int l = i - n_cell[0];
		    int m = j - n_cell[1];
		    int n = k - n_cell[2];
                    Hx_small(l,m,n) = Hx_large(i,j,k);
		    Hy_small(l,m,n) = Hy_large(i,j,k);
		    Hz_small(l,m,n) = Hz_large(i,j,k);
                }
    
                
            });
    }
    */

}





// Function accepts a multifab 'mf' and computes the FFT, storing it in mf_dft_real amd mf_dft_imag multifabs
void ComputeForwardFFT(const MultiFab&    mf,
		       MultiFab&          mf_dft_real,
		       MultiFab&          mf_dft_imag,
		       const Geometry&    geom,
		       long               npts)
{ 
    // **********************************
    // COPY INPUT MULTIFAB INTO A MULTIFAB WITH ONE BOX
    // **********************************

    // create a new BoxArray and DistributionMapping for a MultiFab with 1 grid
    BoxArray ba_onegrid(geom.Domain());
    DistributionMapping dm_onegrid(ba_onegrid);

    // storage for phi and the dft
    MultiFab mf_onegrid         (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_real_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_imag_onegrid(ba_onegrid, dm_onegrid, 1, 0);

    // copy phi into phi_onegrid
    mf_onegrid.ParallelCopy(mf, 0, 0, 1);

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

    // For scaling on forward FFTW
    Real sqrtnpts = std::sqrt(npts);

    // contain to store FFT - note it is shrunk by "half" in x
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;

    Vector<FFTplan> forward_plan;

    for (MFIter mfi(mf_onegrid); mfi.isValid(); ++mfi) {

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
                   mf_onegrid[mfi].dataPtr(),
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   FFTW_ESTIMATE);
#elif (AMREX_SPACEDIM == 3)
      fplan = fftw_plan_dft_r2c_3d(fft_size[2], fft_size[1], fft_size[0],
                   mf_onegrid[mfi].dataPtr(),
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   FFTW_ESTIMATE);
#endif

#endif

      forward_plan.push_back(fplan);
    }

    ParallelDescriptor::Barrier();

    // ForwardTransform
    for (MFIter mfi(mf_onegrid); mfi.isValid(); ++mfi) {
      int i = mfi.LocalIndex();
#ifdef AMREX_USE_CUDA
      cufftSetStream(forward_plan[i], Gpu::gpuStream());
      cufftResult result = cufftExecD2Z(forward_plan[i],
                    mf_onegrid[mfi].dataPtr(),
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
          realpart(i,j,k) /= std::sqrt(npts);
          imagpart(i,j,k) /= std::sqrt(npts);
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


// This function takes the real and imaginary parts of data from the frequency domain and performs an inverse FFT, storing the result in 'mf_2'
// The FFTW c2r function is called which accepts complex data in the frequency domain and returns real data in the normal cartesian plane
void ComputeInverseFFT(MultiFab&                        mf_2,
		       const MultiFab&                  mf_dft_real,
                       const MultiFab&                  mf_dft_imag,				   
		       GpuArray<int, 3>                 n_cell,
                       const Geometry&                  geom)
{

    // create a new BoxArray and DistributionMapping for a MultiFab with 1 grid
    BoxArray ba_onegrid(geom.Domain());
    DistributionMapping dm_onegrid(ba_onegrid);
    
    // Declare multifabs to store entire dataset in one grid.
    MultiFab mf_onegrid_2 (ba_onegrid, dm_onegrid, 1, 0);
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

    // Compute the inverse FFT on spectral_field and store it in 'mf_onegrid_2'
    Vector<FFTplan> backward_plan;

    // Now that we have a spectral field full of the data from the DFT..
    // We perform the inverse DFT on spectral field and store it in mf_onegrid_2
    for (MFIter mfi(mf_onegrid_2); mfi.isValid(); ++mfi) {

       // grab a single box including ghost cell range
       Box realspace_bx = mfi.fabbox();

       // size of box including ghost cell range
       IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

       FFTplan bplan;

#if (AMREX_SPACEDIM == 2)
      bplan = fftw_plan_dft_c2r_2d(fft_size[1], fft_size[0],
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   mf_onegrid_2[mfi].dataPtr(),
                   FFTW_ESTIMATE);
#elif (AMREX_SPACEDIM == 3)
      bplan = fftw_plan_dft_c2r_3d(fft_size[2], fft_size[1], fft_size[0],
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   mf_onegrid_2[mfi].dataPtr(),
                   FFTW_ESTIMATE);
#endif

      backward_plan.push_back(bplan);// This adds an instance of bplan to the end of backward_plan
      }

    for (MFIter mfi(mf_onegrid_2); mfi.isValid(); ++mfi) {
      int i = mfi.LocalIndex();
      fftw_execute(backward_plan[i]);

      // Standard scaling after fft and inverse fft using FFTW
#if (AMREX_SPACEDIM == 2)
      mf_onegrid_2[mfi] /= n_cell[0]*n_cell[1];
#elif (AMREX_SPACEDIM == 3)
      mf_onegrid_2[mfi] /= n_cell[0]*n_cell[1]*n_cell[2];
#endif

    }

    // copy contents of mf_onegrid_2 into mf
    mf_2.ParallelCopy(mf_onegrid_2, 0, 0, 1);

    // destroy ifft plan
    for (int i = 0; i < backward_plan.size(); ++i) {
#ifdef AMREX_USE_CUDA
        cufftDestroy(backward_plan[i]);
#else
        fftw_destroy_plan(backward_plan[i]);
#endif

    }

}

