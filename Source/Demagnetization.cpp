#include "Demagnetization.H"
#include "MagneX.H"
#include "CartesianAlgorithm_K.H"
#include <AMReX_PlotFileUtil.H>

using namespace std;

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
    domain_large.setSmall(dom_lo_large);
    domain_large.setBig  (dom_hi_large);

    // Initialize the boxarray "ba" from the single box "domain"
    ba_large.define(domain_large);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    // create IntVect of max_grid_size (double the value since this is for the large domain)
    IntVect max_grid_size(AMREX_D_DECL(2*max_grid_size_x,2*max_grid_size_y,2*max_grid_size_z));
    ba_large.maxSize(max_grid_size);

    // How Boxes are distributed among MPI processes
    dm_large.define(ba_large);

    // periodic in all directions
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    // This defines a Geometry object
    geom_large.define(domain_large, real_box_large, CoordSys::cartesian, is_periodic);

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo_fft(AMREX_D_DECL(            0,             0,             0));
    IntVect dom_hi_fft(AMREX_D_DECL(n_cell[0], 2*n_cell[1]-1, 2*n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain_fft(dom_lo_fft, dom_hi_fft);

    Box cdomain_large = geom_large.Domain();
    cdomain_large.setBig(0,cdomain_large.length(0)/2);
    Geometry cgeom_large(cdomain_large, real_box_large, CoordSys::cartesian, is_periodic);
    auto cba_large = amrex::decompose(cdomain_large, ParallelContext::NProcsSub(),
                                      {AMREX_D_DECL(true,true,false)});
    DistributionMapping cdm_large(cba_large);

    Kxx_fft.define(cba_large, cdm_large, 1, 0);
    Kxy_fft.define(cba_large, cdm_large, 1, 0);
    Kxz_fft.define(cba_large, cdm_large, 1, 0);
    Kyy_fft.define(cba_large, cdm_large, 1, 0);
    Kyz_fft.define(cba_large, cdm_large, 1, 0);
    Kzz_fft.define(cba_large, cdm_large, 1, 0);

    Mx_fft.define(cba_large, cdm_large, 1, 0);
    My_fft.define(cba_large, cdm_large, 1, 0);
    Mz_fft.define(cba_large, cdm_large, 1, 0);
    Hx_fft.define(cba_large, cdm_large, 1, 0);
    Hy_fft.define(cba_large, cdm_large, 1, 0);
    Hz_fft.define(cba_large, cdm_large, 1, 0);

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
    for (MFIter mfi(Kxx,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

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

    if (plot_int > 0) {

        // Allocate the plot file for the large FFT
        MultiFab Plt (ba_large, dm_large, 6, 0);

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
    }

    amrex_fft = std::make_unique<amrex::FFT::R2C<amrex::Real>>(domain_large);

    amrex_fft->forward(Kxx,Kxx_fft);
    amrex_fft->forward(Kxy,Kxy_fft);
    amrex_fft->forward(Kxz,Kxz_fft);
    amrex_fft->forward(Kyy,Kyy_fft);
    amrex_fft->forward(Kyz,Kyz_fft);
    amrex_fft->forward(Kzz,Kzz_fft);

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

    amrex_fft->forward(Mfield_padded[0], Mx_fft);
    amrex_fft->forward(Mfield_padded[1], My_fft);
    amrex_fft->forward(Mfield_padded[2], Mz_fft);

    for ( MFIter mfi(Kxx_fft,TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.tilebox();

        Array4<GpuComplex<Real>> const& Kxx_fft_ptr = Kxx_fft.array(mfi);
        Array4<GpuComplex<Real>> const& Kxy_fft_ptr = Kxy_fft.array(mfi);
        Array4<GpuComplex<Real>> const& Kxz_fft_ptr = Kxz_fft.array(mfi);
        Array4<GpuComplex<Real>> const& Kyy_fft_ptr = Kyy_fft.array(mfi);
        Array4<GpuComplex<Real>> const& Kyz_fft_ptr = Kyz_fft.array(mfi);
        Array4<GpuComplex<Real>> const& Kzz_fft_ptr = Kzz_fft.array(mfi);

        Array4<GpuComplex<Real>> const& Mx_fft_ptr = Mx_fft.array(mfi);
        Array4<GpuComplex<Real>> const& My_fft_ptr = My_fft.array(mfi);
        Array4<GpuComplex<Real>> const& Mz_fft_ptr = Mz_fft.array(mfi);

        Array4<GpuComplex<Real>> Hx_fft_ptr = Hx_fft.array(mfi);
        Array4<GpuComplex<Real>> Hy_fft_ptr = Hy_fft.array(mfi);
        Array4<GpuComplex<Real>> Hz_fft_ptr = Hz_fft.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Take the dot product in fourier space of M and K and store that in 6 different multifabs
            GpuComplex<Real> Hx_fft_pt(  (Mx_fft_ptr(i,j,k).real() * Kxx_fft_ptr(i,j,k).real() + My_fft_ptr(i,j,k).real() * Kxy_fft_ptr(i,j,k).real() + Mz_fft_ptr(i,j,k).real() * Kxz_fft_ptr(i,j,k).real())
                                       - (Mx_fft_ptr(i,j,k).imag() * Kxx_fft_ptr(i,j,k).imag() + My_fft_ptr(i,j,k).imag() * Kxy_fft_ptr(i,j,k).imag() + Mz_fft_ptr(i,j,k).imag() * Kxz_fft_ptr(i,j,k).imag()),
                                         (Mx_fft_ptr(i,j,k).real() * Kxx_fft_ptr(i,j,k).imag() + My_fft_ptr(i,j,k).real() * Kxy_fft_ptr(i,j,k).imag() + Mz_fft_ptr(i,j,k).real() * Kxz_fft_ptr(i,j,k).imag())
                                       + (Mx_fft_ptr(i,j,k).imag() * Kxx_fft_ptr(i,j,k).real() + My_fft_ptr(i,j,k).imag() * Kxy_fft_ptr(i,j,k).real() + Mz_fft_ptr(i,j,k).imag() * Kxz_fft_ptr(i,j,k).real()) );
            Hx_fft_ptr(i,j,k) = Hx_fft_pt;

            GpuComplex<Real> Hy_fft_pt(  (Mx_fft_ptr(i,j,k).real() * Kxy_fft_ptr(i,j,k).real() + My_fft_ptr(i,j,k).real() * Kyy_fft_ptr(i,j,k).real() + Mz_fft_ptr(i,j,k).real() * Kyz_fft_ptr(i,j,k).real())
                                       - (Mx_fft_ptr(i,j,k).imag() * Kxy_fft_ptr(i,j,k).imag() + My_fft_ptr(i,j,k).imag() * Kyy_fft_ptr(i,j,k).imag() + Mz_fft_ptr(i,j,k).imag() * Kyz_fft_ptr(i,j,k).imag()),
                                         (Mx_fft_ptr(i,j,k).real() * Kxy_fft_ptr(i,j,k).imag() + My_fft_ptr(i,j,k).real() * Kyy_fft_ptr(i,j,k).imag() + Mz_fft_ptr(i,j,k).real() * Kyz_fft_ptr(i,j,k).imag())
                                       + (Mx_fft_ptr(i,j,k).imag() * Kxy_fft_ptr(i,j,k).real() + My_fft_ptr(i,j,k).imag() * Kyy_fft_ptr(i,j,k).real() + Mz_fft_ptr(i,j,k).imag() * Kyz_fft_ptr(i,j,k).real()) );
            Hy_fft_ptr(i,j,k) = Hy_fft_pt;

            GpuComplex<Real> Hz_fft_pt(  (Mx_fft_ptr(i,j,k).real() * Kxz_fft_ptr(i,j,k).real() + My_fft_ptr(i,j,k).real() * Kyz_fft_ptr(i,j,k).real() + Mz_fft_ptr(i,j,k).real() * Kzz_fft_ptr(i,j,k).real())
                                       - (Mx_fft_ptr(i,j,k).imag() * Kxz_fft_ptr(i,j,k).imag() + My_fft_ptr(i,j,k).imag() * Kyz_fft_ptr(i,j,k).imag() + Mz_fft_ptr(i,j,k).imag() * Kzz_fft_ptr(i,j,k).imag()),
                                         (Mx_fft_ptr(i,j,k).real() * Kxz_fft_ptr(i,j,k).imag() + My_fft_ptr(i,j,k).real() * Kyz_fft_ptr(i,j,k).imag() + Mz_fft_ptr(i,j,k).real() * Kzz_fft_ptr(i,j,k).imag())
                                       + (Mx_fft_ptr(i,j,k).imag() * Kxz_fft_ptr(i,j,k).real() + My_fft_ptr(i,j,k).imag() * Kyz_fft_ptr(i,j,k).real() + Mz_fft_ptr(i,j,k).imag() * Kzz_fft_ptr(i,j,k).real()) );
            Hz_fft_ptr(i,j,k) = Hz_fft_pt;
        });
     }

    MultiFab Hx_large(ba_large, dm_large, 1, 0);
    MultiFab Hy_large(ba_large, dm_large, 1, 0);
    MultiFab Hz_large(ba_large, dm_large, 1, 0);

    amrex_fft->backward(Hx_fft,Hx_large);
    amrex_fft->backward(Hy_fft,Hy_large);
    amrex_fft->backward(Hz_fft,Hz_large);

    // scale output inverse of number of cells per FFT convention
    long n_cells = 2*n_cell[0];
    if (AMREX_SPACEDIM >= 2) n_cells *= 2*n_cell[1];
    if (AMREX_SPACEDIM >= 3) n_cells *= 2*n_cell[2];
    Hx_large.mult(1./n_cells);
    Hy_large.mult(1./n_cells);
    Hz_large.mult(1./n_cells);

    // Copying the elements near the 'upper right' of the double-sized demag back to multifab that is the problem size
    // This is not quite the 'upper right' of the source, it's the destination_coordinate + (n_cell-1)
    MultiBlockIndexMapping dtos;
    dtos.offset = IntVect(1-n_cell[0],1-n_cell[1],1-n_cell[2]); // offset = src - dst; "1-n_cell" because we are shifting downward by n_cell-1
    Box dest_box(IntVect(0),IntVect(n_cell[0]-1,n_cell[1]-1,n_cell[2]-1));
    ParallelCopy(H_demagfield[0], dest_box, Hx_large, 0, 0, 1, IntVect(0), dtos);
    ParallelCopy(H_demagfield[1], dest_box, Hy_large, 0, 0, 1, IntVect(0), dtos);
    ParallelCopy(H_demagfield[2], dest_box, Hz_large, 0, 0, 1, IntVect(0), dtos);

}

