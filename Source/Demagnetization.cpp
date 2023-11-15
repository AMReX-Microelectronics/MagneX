#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif

#include <AMReX_PlotFileUtil.H>

#include "Demagnetization.H"
#include "CartesianAlgorithm.H"

void ComputePoissonRHS(MultiFab&                        PoissonRHS,
                       Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                       MultiFab&                        Ms,
                       const Geometry&                  geom)
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
// Then we take the Fourier transform of the demagnetization tensor and return that in 12 different multifabs
void ComputeDemagTensor(MultiFab&                        Kxx_fft_real,
                        MultiFab&                        Kxx_fft_imag,
                        MultiFab&                        Kxy_fft_real,
                        MultiFab&                        Kxy_fft_imag,
                        MultiFab&                        Kxz_fft_real,
                        MultiFab&                        Kxz_fft_imag,
                        MultiFab&                        Kyy_fft_real,
                        MultiFab&                        Kyy_fft_imag,
                        MultiFab&                        Kyz_fft_real,
                        MultiFab&                        Kyz_fft_imag,
                        MultiFab&                        Kzz_fft_real,
                        MultiFab&                        Kzz_fft_imag,
                        GpuArray<int, 3>                 n_cell_large,
                        const Geometry&                  geom_large)
{
    // Extract the domain data 
    BoxArray ba_large = Kxx_fft_real.boxArray();
    DistributionMapping dm_large = Kxx_fft_real.DistributionMap();
	
    // MultiFab storage for the demag tensor
    // TWICE AS BIG AS THE DOMAIN OF THE PROBLEM!!!!!!!!
    MultiFab Kxx (ba_large, dm_large, 1, 0);
    MultiFab Kxy (ba_large, dm_large, 1, 0);
    MultiFab Kxz (ba_large, dm_large, 1, 0);
    MultiFab Kyy (ba_large, dm_large, 1, 0);
    MultiFab Kyz (ba_large, dm_large, 1, 0);
    MultiFab Kzz (ba_large, dm_large, 1, 0);

    Kxx.setVal(0.);
    Kxy.setVal(0.);
    Kxz.setVal(0.);
    Kxz.setVal(0.);
    Kyy.setVal(0.);
    Kyz.setVal(0.);
    Kzz.setVal(0.);

    GpuArray<Real,AMREX_SPACEDIM> dx = geom_large.CellSizeArray();

    // Account for double-sized domain
    dx[0] *= 2.;
    dx[1] *= 2.;
    dx[2] *= 2.;

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
	    if (L == n_cell_large[0]-1 || M == n_cell_large[1]-1 || N == n_cell_large[2]-1){
                return;
            }
	    if (L == n_cell_large[0]/2-1 && M == n_cell_large[1]/2-1 && N == n_cell_large[2]/2-1){
                return;
            }
	    if (L == 0 && M == 0 && N == 0){
	        return;
	    }

            // Need a negative notion of index where demag is centered at the origin, so we make an aritificial copy of it
            int I = L - n_cell_large[0]/2 + 1;
            int J = M - n_cell_large[1]/2 + 1;
            int K = N - n_cell_large[2]/2 + 1;

            // **********************************
            // SET VALUES FOR EACH CELL
            // **********************************
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

            Kxx_ptr(L,M,N) *= prefactor;
            Kxy_ptr(L,M,N) *= (-prefactor);
            Kxz_ptr(L,M,N) *= (-prefactor);
            Kyy_ptr(L,M,N) *= prefactor;
            Kyz_ptr(L,M,N) *= (-prefactor);
            Kzz_ptr(L,M,N) *= prefactor;

        });
    }

    ComputeForwardFFT(Kxx, Kxx_fft_real, Kxx_fft_imag, geom_large);
    ComputeForwardFFT(Kxy, Kxy_fft_real, Kxy_fft_imag, geom_large);
    ComputeForwardFFT(Kxz, Kxz_fft_real, Kxz_fft_imag, geom_large);
    ComputeForwardFFT(Kyy, Kyy_fft_real, Kyy_fft_imag, geom_large);
    ComputeForwardFFT(Kyz, Kyz_fft_real, Kyz_fft_imag, geom_large);
    ComputeForwardFFT(Kzz, Kzz_fft_real, Kzz_fft_imag, geom_large);
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
void ComputeHFieldFFT(const Array<MultiFab, AMREX_SPACEDIM>& M_field_padded,
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
                      GpuArray<int, 3>                       n_cell_large,
                      const Geometry&                        geom_large)
{
    BoxArray ba_large = Kxx_fft_real.boxArray();
    DistributionMapping dm_large = Kxx_fft_real.DistributionMap();

    // Allocate M_field fft multifabs
    MultiFab M_dft_real_x(ba_large, dm_large, 1, 0);
    MultiFab M_dft_imag_x(ba_large, dm_large, 1, 0);
    MultiFab M_dft_real_y(ba_large, dm_large, 1, 0);
    MultiFab M_dft_imag_y(ba_large, dm_large, 1, 0);
    MultiFab M_dft_real_z(ba_large, dm_large, 1, 0);
    MultiFab M_dft_imag_z(ba_large, dm_large, 1, 0);

    // Calculate the Mx, My, and Mz fft's at the current time step
    // Each fft will be stored in seperate real and imaginary multifabs
    ComputeForwardFFT(M_field_padded[0], M_dft_real_x, M_dft_imag_x, geom_large);
    ComputeForwardFFT(M_field_padded[1], M_dft_real_y, M_dft_imag_y, geom_large);
    ComputeForwardFFT(M_field_padded[2], M_dft_real_z, M_dft_imag_z, geom_large);

    // Allocate 6 Multifabs to store the convolutions in Fourier space for H_field
    // This could be done in main but then we have an insane amount of arguments in this function
    MultiFab H_dft_real_x(ba_large, dm_large, 1, 0);
    MultiFab H_dft_imag_x(ba_large, dm_large, 1, 0);
    MultiFab H_dft_real_y(ba_large, dm_large, 1, 0);
    MultiFab H_dft_imag_y(ba_large, dm_large, 1, 0);
    MultiFab H_dft_real_z(ba_large, dm_large, 1, 0);
    MultiFab H_dft_imag_z(ba_large, dm_large, 1, 0);

    for ( MFIter mfi(M_field_padded[0]); mfi.isValid(); ++mfi )
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

    // Allocate Multifabs to store large H_fieldi
    MultiFab Hx_large(ba_large, dm_large, 1, 0);
    MultiFab Hy_large(ba_large, dm_large, 1, 0);
    MultiFab Hz_large(ba_large, dm_large, 1, 0);

    // Compute the inverse FFT of H_field with respect to the three coordinates and store them in 3 multifabs that this function returns
    ComputeInverseFFT(Hx_large, H_dft_real_x, H_dft_imag_x, n_cell_large, geom_large);
    ComputeInverseFFT(Hy_large, H_dft_real_y, H_dft_imag_y, n_cell_large, geom_large);
    ComputeInverseFFT(Hz_large, H_dft_real_z, H_dft_imag_z, n_cell_large, geom_large); 

    // create a new BoxArray and DistributionMapping for a MultiFab with 1 grid
    BoxArray ba_onegrid(geom_large.Domain());
    DistributionMapping dm_onegrid(ba_onegrid);

    // Storage for the double-sized Hfield on 1 grid 
    MultiFab Hx_large_onegrid (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab Hy_large_onegrid (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab Hz_large_onegrid (ba_onegrid, dm_onegrid, 1, 0);

    // Copy the distributed Hfield multifabs into onegrid multifabs
    Hx_large_onegrid.ParallelCopy(Hx_large, 0, 0, 1);
    Hy_large_onegrid.ParallelCopy(Hy_large, 0, 0, 1);
    Hz_large_onegrid.ParallelCopy(Hz_large, 0, 0, 1);

    // Storage for the small 1 grid Hfield
    MultiFab Hx_small_onegrid (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab Hy_small_onegrid (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab Hz_small_onegrid (ba_onegrid, dm_onegrid, 1, 0);

    // Copying the elements in the 'upper right'  of the double-sized demag back to multifab that is the problem size
    for ( MFIter mfi(Hx_small_onegrid); mfi.isValid(); ++mfi )
    {
            const Box& bx = mfi.validbox();

            const Array4<Real>& Hx_large_onegrid_ptr = Hx_large_onegrid.array(mfi);
            const Array4<Real>& Hy_large_onegrid_ptr = Hy_large_onegrid.array(mfi);
            const Array4<Real>& Hz_large_onegrid_ptr = Hz_large_onegrid.array(mfi);

	    const Array4<Real>& Hx_small_onegrid_ptr = Hx_small_onegrid.array(mfi);
            const Array4<Real>& Hy_small_onegrid_ptr = Hy_small_onegrid.array(mfi);
            const Array4<Real>& Hz_small_onegrid_ptr = Hz_small_onegrid.array(mfi);

  
  	    amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (i >= ((n_cell_large[0]/2)-1) && j >= ((n_cell_large[1]/2)-1) && k >= ((n_cell_large[2]/2)-1)){
		    if (i >= n_cell_large[0]-1 || j >= n_cell_large[1]-1 || k >= n_cell_large[2]-1){
                        return;
                    }
		    int l = i - n_cell_large[0]/2 + 1;
		    int m = j - n_cell_large[1]/2 + 1;
		    int n = k - n_cell_large[2]/2 + 1;
                    Hx_small_onegrid_ptr(l,m,n) = Hx_large_onegrid_ptr(i,j,k);
		    Hy_small_onegrid_ptr(l,m,n) = Hy_large_onegrid_ptr(i,j,k);
		    Hz_small_onegrid_ptr(l,m,n) = Hz_large_onegrid_ptr(i,j,k);
                }
    
                
            });
    }

    // Store the final result in the distributed array of multifabs
    H_demagfield[0].ParallelCopy(Hx_small_onegrid, 0, 0, 1);
    H_demagfield[1].ParallelCopy(Hy_small_onegrid, 0, 0, 1);
    H_demagfield[2].ParallelCopy(Hz_small_onegrid, 0, 0, 1);
    

}

// Function accepts a multifab 'mf' and computes the FFT, storing it in mf_dft_real amd mf_dft_imag multifabs
void ComputeForwardFFT(const MultiFab&    mf,
		       MultiFab&          mf_dft_real,
		       MultiFab&          mf_dft_imag,
		       const Geometry&    geom)
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
                   mf_onegrid_2[mfi].dataPtr(),
                   FFTW_ESTIMATE);
#elif (AMREX_SPACEDIM == 3)
      bplan = fftw_plan_dft_c2r_3d(fft_size[2], fft_size[1], fft_size[0],
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   mf_onegrid_2[mfi].dataPtr(),
                   FFTW_ESTIMATE);
#endif

#endif

      backward_plan.push_back(bplan);// This adds an instance of bplan to the end of backward_plan
      }

    for (MFIter mfi(mf_onegrid_2); mfi.isValid(); ++mfi) {
      int i = mfi.LocalIndex();

#ifdef AMREX_USE_CUDA
      cufftSetStream(backward_plan[i], Gpu::gpuStream());
      cufftResult result = cufftExecZ2D(backward_plan[i],
                           reinterpret_cast<FFTcomplex*>
                           (spectral_field[i]->dataPtr()),
                           mf_onegrid_2[mfi].dataPtr());
       if (result != CUFFT_SUCCESS) {
         AllPrint() << " inverse transform using cufftExec failed! Error: "
         << cufftErrorToString(result) << "\n";
       }
#else
      fftw_execute(backward_plan[i]);
#endif

    }

      // Standard scaling after fft and inverse fft using FFTW
#if (AMREX_SPACEDIM == 2)
    mf_onegrid_2.mult(1./(n_cell[0]*n_cell[1]));
#elif (AMREX_SPACEDIM == 3)
    mf_onegrid_2.mult(1./(n_cell[0]*n_cell[1]*n_cell[2]));
#endif
    
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

#ifdef AMREX_USE_CUDA
std::string cufftErrorToString (const cufftResult& err)
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
