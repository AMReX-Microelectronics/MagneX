#include "MagnetostaticSolver.H"
#include "CartesianAlgorithm.H"

void ComputePoissonRHS(MultiFab&                        PoissonRHS,
                       //Array<MultiFab, AMREX_SPACEDIM>& Mfield,
		       const amrex::Vector<MultiFab>& Mfield,
                       Array<MultiFab, AMREX_SPACEDIM>& Ms,
                       const Geometry&                 geom)
{
    for ( MFIter mfi(PoissonRHS); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real const>& M_xface = Mfield[0].array(mfi);         
            const Array4<Real const>& M_yface = Mfield[1].array(mfi);         
            const Array4<Real const>& M_zface = Mfield[2].array(mfi);   

            const Array4<Real>& rhs = PoissonRHS.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

               rhs(i,j,k) =  DivergenceDx_Mag(M_xface, i, j, k, dx, 0)
                           + DivergenceDy_Mag(M_yface, i, j, k, dx, 1)
                           + DivergenceDz_Mag(M_zface, i, j, k, dx, 2);
                
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

