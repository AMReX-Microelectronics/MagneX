#include "MagneX.H"
#include "CartesianAlgorithm_K.H"
#include <cmath>

long CountMagneticCells(MultiFab& Ms)
{
    // timer for profiling
    BL_PROFILE_VAR("CountMagneticCells()",CountMagneticCells);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<long> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(Ms,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto const& fab = Ms.array(mfi);

        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (fab(i,j,k) > 0.) {
                return {1};
            } else {
                return {0};
            }
        });
    }

    long sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceLongSum(sum);

    return sum;
}

Real SumNormalizedM(MultiFab& Ms,
                    MultiFab& Mfield)
{
    // timer for profiling
    BL_PROFILE_VAR("SumNormalizedM()",SumNormalizedM);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(Ms,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto const& fab = Ms.array(mfi);
        auto const& M = Mfield.array(mfi);
        
	reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (fab(i,j,k) > 0.) {
                return {M(i,j,k)/fab(i,j,k)};
            } else {
                return {0.};
            }
	});
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
}

Real SumHbias(MultiFab& H_biasfield,
	      MultiFab& Ms)
{
    // timer for profiling
    BL_PROFILE_VAR("SumNormalizedM()",SumNormalizedM);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(H_biasfield,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        // extract field data
        auto const& H_bias = H_biasfield.array(mfi);
        auto const& fab = Ms.array(mfi);

        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
	    if (fab(i,j,k)> 0.){
                return {H_bias(i,j,k)/fab(i,j,k)};
            } else {
                return {0.};
	    }
	});
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
}

Real DemagEnergy(MultiFab& Ms,
                  MultiFab& Mfield_x,
                  MultiFab& Mfield_y,
                  MultiFab& Mfield_z,
                  MultiFab& Demagfield_x,
		  MultiFab& Demagfield_y,
		  MultiFab& Demagfield_z)
{
    ReduceOps<ReduceOpSum> reduce_op;

    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(Ms,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto const& fab = Ms.array(mfi);

        auto const& Mx = Mfield_x.array(mfi);
        auto const& My = Mfield_y.array(mfi);
        auto const& Mz = Mfield_z.array(mfi);

        auto const& demag_x = Demagfield_x.array(mfi);
        auto const& demag_y = Demagfield_y.array(mfi);
        auto const& demag_z = Demagfield_z.array(mfi);

        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (fab(i,j,k) > 0.) {
                return {((Mx(i,j,k))*(demag_x(i,j,k)) + (My(i,j,k))*(demag_y(i,j,k)) + (Mz(i,j,k))*(demag_z(i,j,k)))};
            } else {
                return {0.};
            }
        });
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum * (-mu0/2.);
}

Real ExchangeEnergy(Array< MultiFab, AMREX_SPACEDIM>& Mfield,
		    MultiFab& Ms,
		    const Geometry& geom,
                    Real exch_const)
    {
    // timer for profiling
    BL_PROFILE_VAR("CalculateH_exchange()",CalculateH_exchange);

    ReduceOps<ReduceOpSum> reduce_op;

    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;
	
    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // extract dd from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dd = geom.CellSizeArray();	   
    
	const Box& bx = mfi.validbox();

        auto const& Mx = Mfield[0].array(mfi);
        auto const& My = Mfield[1].array(mfi);
        auto const& Mz = Mfield[2].array(mfi);
        auto const& Ms_arr = Ms.array(mfi);

	reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
	{

 	    // determine if the material is magnetic or not
                if (Ms_arr(i,j,k) > 0.){

                    // H_exchange - use M^(old_time)
                    // amrex::Real const H_exchange_coeff = 2.0 * exchange_arr(i,j,k) / mu0 / Ms_arr(i,j,k) / Ms_arr(i,j,k);
                    // Neumann boundary condition dM/dn = -1/xi (z x n) x M
                    // amrex::Real xi_DMI = 0.0; // xi_DMI cannot be zero, this is just initialization

                    amrex::Real Ms_lo_x = Ms_arr(i-1, j, k);
                    amrex::Real Ms_hi_x = Ms_arr(i+1, j, k);
                    amrex::Real Ms_lo_y = Ms_arr(i, j-1, k);
                    amrex::Real Ms_hi_y = Ms_arr(i, j+1, k);
                    amrex::Real Ms_lo_z = Ms_arr(i, j, k-1);
                    amrex::Real Ms_hi_z = Ms_arr(i, j, k+1);

                    // // Neumann boundary condition in scalar form, dMx/dx = -/+ 1/xi*Mz 
                    // at x-faces, dM/dn = dM/dx
                    amrex::Real dMxdx_BC_lo_x = 0.0; // lower x BC: dMx/dx = 1/xi*Mz
                    amrex::Real dMxdx_BC_hi_x = 0.0; // higher x BC: dMx/dx = -1/xi*Mz
                    amrex::Real dMxdy_BC_lo_y = 0.0;
                    amrex::Real dMxdy_BC_hi_y = 0.0;
                    amrex::Real dMxdz_BC_lo_z = 0.0;
                    amrex::Real dMxdz_BC_hi_z = 0.0;

                    amrex::Real dMydx_BC_lo_x = 0.0;
                    amrex::Real dMydx_BC_hi_x = 0.0;
                    amrex::Real dMydy_BC_lo_y = 0.0; // if with DMI, lower y BC: dMy/dy = 1/xi*Mz
                    amrex::Real dMydy_BC_hi_y = 0.0; // if with DMI, higher y BC: dMy/dy = -1/xi*Mz
                    amrex::Real dMydz_BC_lo_z = 0.0;
                    amrex::Real dMydz_BC_hi_z = 0.0;

                    amrex::Real dMzdx_BC_lo_x = 0.0; // if with DMI, lower x BC: dMz/dx = -1/xi*Mx
                    amrex::Real dMzdx_BC_hi_x = 0.0; // if with DMI, higher x BC: dMz/dx = 1/xi*Mx
                    amrex::Real dMzdy_BC_lo_y = 0.0; // if with DMI, lower y BC: dMz/dy = -1/xi*My
                    amrex::Real dMzdy_BC_hi_y = 0.0; // if with DMI, higher y BC: dMz/dy = 1/xi*My
                    amrex::Real dMzdz_BC_lo_z = 0.0; // dMz/dz = 0
                    amrex::Real dMzdz_BC_hi_z = 0.0; // dMz/dz = 0

		    // Take 9 spacial derivatives where 'Hxy' is the derivative of Mx with respect to y
		    amrex::Real Hxx = DMDx_Mag(Mx, Ms_lo_x, Ms_hi_x, dMxdx_BC_lo_x, dMxdx_BC_hi_x, i, j, k, dd);
		    amrex::Real Hxy = DMDy_Mag(Mx, Ms_lo_y, Ms_hi_y, dMydx_BC_lo_x, dMydx_BC_hi_x, i, j, k, dd);
		    amrex::Real Hxz = DMDz_Mag(Mx, Ms_lo_z, Ms_hi_z, dMzdx_BC_lo_x, dMzdx_BC_hi_x, i, j, k, dd);

                    amrex::Real Hyx = DMDx_Mag(My, Ms_lo_x, Ms_hi_x, dMxdy_BC_lo_y, dMxdy_BC_hi_y, i, j, k, dd); 
		    amrex::Real Hyy = DMDy_Mag(My, Ms_lo_y, Ms_hi_y, dMydy_BC_lo_y, dMydy_BC_hi_y, i, j, k, dd); 
		    amrex::Real Hyz = DMDz_Mag(My, Ms_lo_z, Ms_hi_z, dMzdy_BC_lo_y, dMzdy_BC_hi_y, i, j, k, dd);			
		    
		    amrex::Real Hzx = DMDx_Mag(Mz, Ms_lo_x, Ms_hi_x, dMxdz_BC_lo_z, dMxdz_BC_hi_z, i, j, k, dd);
		    amrex::Real Hzy = DMDy_Mag(Mz, Ms_lo_y, Ms_hi_y, dMydz_BC_lo_z, dMydz_BC_hi_z, i, j, k, dd);
		    amrex::Real Hzz = DMDz_Mag(Mz, Ms_lo_z, Ms_hi_z, dMzdz_BC_lo_z, dMzdz_BC_hi_z, i, j, k, dd);
                         
			/*
                        if (DMI_coupling == 1) {
                            if (DMI_arr(i,j,k) == 0.) amrex::Abort("The DMI_arr(i,j,k) is 0.0 while including the DMI coupling");

                            xi_DMI = 2.0*exchange_arr(i,j,k)/DMI_arr(i,j,k);

                            dMxdx_BC_lo_x = -1.0/xi_DMI*Mz(i,j,k) ; // lower x BC: dMx/dx = 1/xi*Mz
                            dMxdx_BC_hi_x = -1.0/xi_DMI*Mz(i,j,k) ; // higher x BC: dMx/dx = -1/xi*Mz

                            dMydy_BC_lo_y = -1.0/xi_DMI*Mz(i,j,k) ; // lower y BC: dMy/dy = 1/xi*Mz
                            dMydy_BC_hi_y = -1.0/xi_DMI*Mz(i,j,k) ; // higher y BC: dMy/dy = -1/xi*Mz

                            dMzdx_BC_lo_x =  1.0/xi_DMI*Mx(i,j,k);  // lower x BC: dMz/dx = -1/xi*Mx
                            dMzdx_BC_hi_x =  1.0/xi_DMI*Mx(i,j,k);  // higher x BC: dMz/dx = 1/xi*Mx
                            dMzdy_BC_lo_y =  1.0/xi_DMI*My(i,j,k);  // lower y BC: dMz/dy = -1/xi*My
                            dMzdy_BC_hi_y =  1.0/xi_DMI*My(i,j,k);  // higher y BC: dMz/dy = 1/xi*My
                        }
                        */
			
		    return{std::pow(Hxx/Ms_arr(i,j,k),2) + std::pow(Hxy/Ms_arr(i,j,k),2) + std::pow( Hxz/Ms_arr(i,j,k),2) + std::pow(Hyx/Ms_arr(i,j,k),2) + std::pow(Hyy/Ms_arr(i,j,k),2) + std::pow(Hyz/Ms_arr(i,j,k),2) + std::pow(Hzx/Ms_arr(i,j,k),2) + std::pow(Hzy/Ms_arr(i,j,k),2) + std::pow(Hzz/Ms_arr(i,j,k),2)};
		
		} else {
                    return{0.};
                }
            });
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum * exch_const;
}

Real AnisotropyEnergy(MultiFab& Ms,
                      MultiFab& Mfield_x,
                      MultiFab& Mfield_y,
                      MultiFab& Mfield_z,
		      Real anis)
{
    // timer for profiling
    // BL_PROFILE_VAR("SumNormalizedM()",SumNormalizedM);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(Ms,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto const& fab = Ms.array(mfi);
        auto const& Mx = Mfield_x.array(mfi);
        auto const& My = Mfield_y.array(mfi);
        auto const& Mz = Mfield_z.array(mfi);

        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (fab(i,j,k) > 0.) {
	        return {1.-(std::pow(((Mx(i,j,k)/fab(i,j,k))*anisotropy_axis[0] + (My(i,j,k)/fab(i,j,k))*anisotropy_axis[1] + (Mz(i,j,k)/fab(i,j,k))*anisotropy_axis[2]), 2))};

    	    } else {
                return {0.};
            }
        });
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum * (anis);
}
