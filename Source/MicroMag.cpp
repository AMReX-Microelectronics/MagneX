#include "MicroMag.H"
#include "MagLaplacian.H"

void InitializeMagneticProperties(std::array< MultiFab, AMREX_SPACEDIM >&  alpha,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
                   std::array< MultiFab, AMREX_SPACEDIM >&   gamma,
                   std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
                   std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
                   Real        alpha_val,
                   Real        Ms_val,
                   Real        gamma_val,
                   Real        exchange_val,
		             Real        anisotropy_val,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   amrex::GpuArray<amrex::Real, 3> mag_lo,
                   amrex::GpuArray<amrex::Real, 3> mag_hi,
                   const       Geometry& geom)
{

    for(int i = 0; i < 3; i++){
       alpha[i].setVal(0.);
       Ms[i].setVal(0.);
       gamma[i].setVal(0.);
       exchange[i].setVal(0.);
       anisotropy[i].setVal(0.);
    }

    // loop over boxes
    for (MFIter mfi(alpha[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        Box const &tbx = mfi.tilebox(alpha[0].ixType().toIntVect());
        Box const &tby = mfi.tilebox(alpha[1].ixType().toIntVect());
        Box const &tbz = mfi.tilebox(alpha[2].ixType().toIntVect());

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
        
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

        //xface
        amrex::ParallelFor( tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + i * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
             
            if (x > mag_lo[0] && x < mag_hi[0]){
               if (y > mag_lo[1] && y < mag_hi[1]){
                  if (z > mag_lo[2] && z < mag_hi[2]){
                     alpha_xface_arr(i,j,k) = alpha_val;
                     gamma_xface_arr(i,j,k) = gamma_val;
                     Ms_xface_arr(i,j,k) = Ms_val;
                     exchange_xface_arr(i,j,k) = exchange_val;
                     anisotropy_xface_arr(i,j,k) = anisotropy_val;
                  }
               }
            }
        }); 

        //yface
        amrex::ParallelFor( tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + j * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
             
            if (x > mag_lo[0] && x < mag_hi[0]){
               if (y > mag_lo[1] && y < mag_hi[1]){
                  if (z > mag_lo[2] && z < mag_hi[2]){
                     alpha_yface_arr(i,j,k) = alpha_val;
                     gamma_yface_arr(i,j,k) = gamma_val;
                     Ms_yface_arr(i,j,k) = Ms_val;
                     exchange_yface_arr(i,j,k) = exchange_val;
                     anisotropy_yface_arr(i,j,k) = anisotropy_val;
                  }
               }
            }
        }); 

        //zface
        amrex::ParallelFor( tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + k * dx[2];
             
            if (x > mag_lo[0] && x < mag_hi[0]){
               if (y > mag_lo[1] && y < mag_hi[1]){
                  if (z > mag_lo[2] && z < mag_hi[2]){
                     alpha_zface_arr(i,j,k) = alpha_val;
                     gamma_zface_arr(i,j,k) = gamma_val;
                     Ms_zface_arr(i,j,k) = Ms_val;
                     exchange_zface_arr(i,j,k) = exchange_val;
                     anisotropy_zface_arr(i,j,k) = anisotropy_val;
                  }
               }
            }
        }); 
     }
//     // fill periodic ghost cells
//    for(int i = 0; i < 3; i++){
//       alpha[i].FillBoundary(geom.periodicity());
//       Ms[i].FillBoundary(geom.periodicity());
//       gamma[i].FillBoundary(geom.periodicity());
//       exchange[i].FillBoundary(geom.periodicity());
//       anisotropy[i].FillBoundary(geom.periodicity());
//    }
//
} 

void ComputePoissonRHS(MultiFab&                        PoissonRHS,
                       Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                       Array<MultiFab, AMREX_SPACEDIM>& Ms,
                       const Geometry&                 geom)
{
    for ( MFIter mfi(PoissonRHS); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            Array4<Real> const &M_xface = Mfield[0].array(mfi);         
            Array4<Real> const &M_yface = Mfield[1].array(mfi);         
            Array4<Real> const &M_zface = Mfield[2].array(mfi);   
            const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
            const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
            const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);
            const Array4<Real>& rhs = PoissonRHS.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

               rhs(i,j,k) = DivergenceDx_Mag(M_xface, i, j, k, dx, 0)
                           + DivergenceDy_Mag(M_yface, i, j, k, dx, 1)
                           + DivergenceDz_Mag(M_zface, i, j, k, dx, 2);
                
            });
        }
   
}

void ComputeHfromPhi(MultiFab&                         PoissonPhi,
                      Array<MultiFab, AMREX_SPACEDIM>& H_demagfield,
                      amrex::GpuArray<amrex::Real, 3> prob_lo,
                      amrex::GpuArray<amrex::Real, 3> prob_hi,
                      const Geometry&                 geom)
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
                     Hx_demag(i,j,k) = -(phi(i+1,j,k) - phi(i,j,k))/(dx[0]);

                     Hy_demag(i,j,k) = -(phi(i,j+1,k) - phi(i,j,k))/(dx[1]);

                     Hz_demag(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k))/(dx[2]);
             });
        }

}

