#include "MicroMag.H"
#include "MagLaplacian.H"

void InitializeMagneticProperties(MultiFab&  alpha,
                                  MultiFab&   Ms,
                                  MultiFab&   gamma,
                                  MultiFab&   exchange,
                                  MultiFab&   anisotropy,
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

    alpha.setVal(0.);
    Ms.setVal(0.);
    gamma.setVal(0.);
    exchange.setVal(0.);
    anisotropy.setVal(0.);

    // loop over boxes
    for (MFIter mfi(alpha); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
        
        const Array4<Real>& alpha_arr = alpha.array(mfi);
        const Array4<Real>& gamma_arr = gamma.array(mfi);
        const Array4<Real>& Ms_arr = Ms.array(mfi);
        const Array4<Real>& exchange_arr = exchange.array(mfi);
        const Array4<Real>& anisotropy_arr = anisotropy.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
             
            if (x > mag_lo[0] && x < mag_hi[0]){
               if (y > mag_lo[1] && y < mag_hi[1]){
                  if (z > mag_lo[2] && z < mag_hi[2]){
                     alpha_arr(i,j,k) = alpha_val;
                     gamma_arr(i,j,k) = gamma_val;
                     Ms_arr(i,j,k) = Ms_val;
                     exchange_arr(i,j,k) = exchange_val;
                     anisotropy_arr(i,j,k) = anisotropy_val;
                  }
               }
            }
        }); 
     }
     // fill periodic ghost cells
     alpha.FillBoundary(geom.periodicity());
     Ms.FillBoundary(geom.periodicity());
     gamma.FillBoundary(geom.periodicity());
     exchange.FillBoundary(geom.periodicity());
     anisotropy.FillBoundary(geom.periodicity());

} 

void ComputePoissonRHS(MultiFab&                       PoissonRHS,
                       Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                       MultiFab&                       Ms,
                       const Geometry&                 geom)
{
    for ( MFIter mfi(PoissonRHS); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real>& Mx = Mfield[0].array(mfi);
            const Array4<Real>& My = Mfield[1].array(mfi);
            const Array4<Real>& Mz = Mfield[2].array(mfi);
            const Array4<Real>& Ms_arr = Ms.array(mfi);
            const Array4<Real>& rhs = PoissonRHS.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real Ms_lo_x = Ms_arr(i-1, j, k); 
                amrex::Real Ms_hi_x = Ms_arr(i+1, j, k); 
                amrex::Real Ms_lo_y = Ms_arr(i, j-1, k); 
                amrex::Real Ms_hi_y = Ms_arr(i, j+1, k); 
                amrex::Real Ms_lo_z = Ms_arr(i, j, k-1);
                amrex::Real Ms_hi_z = Ms_arr(i, j, k+1);

                if (Ms_arr(i,j,k) > 0._rt) {
                    rhs(i,j,k) = DivergenceDx_Mag(Mx, Ms_lo_x, Ms_hi_x, i, j, k, dx)
                               + DivergenceDy_Mag(My, Ms_lo_y, Ms_hi_y, i, j, k, dx)
                               + DivergenceDz_Mag(Mz, Ms_lo_z, Ms_hi_z, i, j, k, dx);
                } else {
                    rhs(i,j,k) = 0.0;
                }
                if (i == 128 && j == 32 && k == 4){
                    printf("RHS for Poisson = %g \n", rhs(i,j,k));
                    printf("Ms = %g \n", Ms_arr(i,j,k));
                }
            });
        }
   
}

void ComputeHfromPhi(MultiFab&                        PoissonPhi,
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

            const Array4<Real>& Hx_arr = H_demagfield[0].array(mfi);
            const Array4<Real>& Hy_arr = H_demagfield[1].array(mfi);
            const Array4<Real>& Hz_arr = H_demagfield[2].array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                     Real x    = prob_lo[0] + (i+0.5) * dx[0];
                     Real x_hi = prob_lo[0] + (i+1.5) * dx[0];
                     Real x_lo = prob_lo[0] + (i-0.5) * dx[0];
                     Real y    = prob_lo[1] + (j+0.5) * dx[1];
                     Real y_hi = prob_lo[1] + (j+1.5) * dx[1];
                     Real y_lo = prob_lo[1] + (j-0.5) * dx[1];
                     Real z    = prob_lo[2] + (k+0.5) * dx[2];
                     Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                     Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

                     if(x_lo < prob_lo[0]){ //Bottom Boundary
                       Hx_arr(i,j,k) = -(phi(i+1,j,k) - phi(i,j,k))/(dx[0]); // 1st order accuracy at the outer boundary
                     } else if (x_hi > prob_hi[0]){ //Top Boundary
                       Hx_arr(i,j,k) = -(phi(i,j,k) - phi(i-1,j,k))/(dx[0]); // 1st order accuracy at the outer boundary
                     } else{ //inside
                       Hx_arr(i,j,k) = -(phi(i+1,j,k) - phi(i-1,j,k))/(2.*dx[0]);
                     }

                     if(y_lo < prob_lo[1]){ //Bottom Boundary
                       Hy_arr(i,j,k) = -(phi(i,j+1,k) - phi(i,j,k))/(dx[1]); // 1st order accuracy at the outer boundary
                     } else if (y_hi > prob_hi[1]){ //Top Boundary
                       Hy_arr(i,j,k) = -(phi(i,j,k) - phi(i,j-1,k))/(dx[1]); // 1st order accuracy at the outer boundary
                     } else{ //inside
                       Hy_arr(i,j,k) = -(phi(i,j+1,k) - phi(i,j-1,k))/(2.*dx[1]);
                     }

                     if(z_lo < prob_lo[2]){ //Bottom Boundary
                       Hz_arr(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k))/(dx[2]); // 1st order accuracy at the outer boundary
                     } else if (z_hi > prob_hi[2]){ //Top Boundary
                       Hz_arr(i,j,k) = -(phi(i,j,k) - phi(i,j,k-1))/(dx[2]); // 1st order accuracy at the outer boundary
                     } else{ //inside
                       Hz_arr(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k-1))/(2.*dx[2]);
                     }
                     if (i == 128 && j == 32 && k == 4){
                        printf("Hx_demag = %g \n", Hx_arr(i,j,k));
                     }
             });
        }

}

