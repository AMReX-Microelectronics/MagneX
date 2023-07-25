#include "Diagnostics.H"

void mf_avg_fc_to_cc(MultiFab&  Plt,
                   //std::array< MultiFab, AMREX_SPACEDIM >&   Mfield,
                    amrex::Vector<MultiFab>& Mfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_exchangefield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_DMIfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_anisotropyfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms)
{
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

            const Array4<Real>& H_exchange_xface = H_exchangefield[0].array(mfi);         
            const Array4<Real>& H_exchange_yface = H_exchangefield[1].array(mfi);         
            const Array4<Real>& H_exchange_zface = H_exchangefield[2].array(mfi);

            const Array4<Real>& H_DMI_xface = H_DMIfield[0].array(mfi);         
            const Array4<Real>& H_DMI_yface = H_DMIfield[1].array(mfi);         
            const Array4<Real>& H_DMI_zface = H_DMIfield[2].array(mfi);

            const Array4<Real>& H_anisotropy_xface = H_anisotropyfield[0].array(mfi);         
            const Array4<Real>& H_anisotropy_yface = H_anisotropyfield[1].array(mfi);         
            const Array4<Real>& H_anisotropy_zface = H_anisotropyfield[2].array(mfi);
          
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

                //Hx_exchange at xface, yface, zface
                plt(i,j,k,21) = 0.5 * ( H_exchange_xface(i,j,k,0) + H_exchange_xface(i+1,j,k,0) );   
                plt(i,j,k,22) = 0.5 * ( H_exchange_yface(i,j,k,0) + H_exchange_yface(i,j+1,k,0) );   
                plt(i,j,k,23) = 0.5 * ( H_exchange_zface(i,j,k,0) + H_exchange_zface(i,j,k+1,0) );  
 
                //Hy_exchange at xface, yface, zface
                plt(i,j,k,24) = 0.5 * ( H_exchange_xface(i,j,k,1) + H_exchange_xface(i+1,j,k,1) );   
                plt(i,j,k,25) = 0.5 * ( H_exchange_yface(i,j,k,1) + H_exchange_yface(i,j+1,k,1) );   
                plt(i,j,k,26) = 0.5 * ( H_exchange_zface(i,j,k,1) + H_exchange_zface(i,j,k+1,1) );  
 
                //Hz_exchange at xface, yface, zface
                plt(i,j,k,27) = 0.5 * ( H_exchange_xface(i,j,k,2) + H_exchange_xface(i+1,j,k,2) );   
                plt(i,j,k,28) = 0.5 * ( H_exchange_yface(i,j,k,2) + H_exchange_yface(i,j+1,k,2) );   
                plt(i,j,k,29) = 0.5 * ( H_exchange_zface(i,j,k,2) + H_exchange_zface(i,j,k+1,2) );  

                //Hx_DMI at xface, yface, zface
                plt(i,j,k,30) = 0.5 * ( H_DMI_xface(i,j,k,0) + H_DMI_xface(i+1,j,k,0) );   
                plt(i,j,k,31) = 0.5 * ( H_DMI_yface(i,j,k,0) + H_DMI_yface(i,j+1,k,0) );   
                plt(i,j,k,32) = 0.5 * ( H_DMI_zface(i,j,k,0) + H_DMI_zface(i,j,k+1,0) );  
 
                //Hy_DMI at xface, yface, zface
                plt(i,j,k,33) = 0.5 * ( H_DMI_xface(i,j,k,1) + H_DMI_xface(i+1,j,k,1) );   
                plt(i,j,k,34) = 0.5 * ( H_DMI_yface(i,j,k,1) + H_DMI_yface(i,j+1,k,1) );   
                plt(i,j,k,35) = 0.5 * ( H_DMI_zface(i,j,k,1) + H_DMI_zface(i,j,k+1,1) );  
 
                //Hz_DMI at xface, yface, zface
                plt(i,j,k,36) = 0.5 * ( H_DMI_xface(i,j,k,2) + H_DMI_xface(i+1,j,k,2) );   
                plt(i,j,k,37) = 0.5 * ( H_DMI_yface(i,j,k,2) + H_DMI_yface(i,j+1,k,2) );   
                plt(i,j,k,38) = 0.5 * ( H_DMI_zface(i,j,k,2) + H_DMI_zface(i,j,k+1,2) ); 

                //Hx_anisotropy at xface, yface, zface
                plt(i,j,k,39) = 0.5 * ( H_anisotropy_xface(i,j,k,0) + H_anisotropy_xface(i+1,j,k,0) );   
                plt(i,j,k,40) = 0.5 * ( H_anisotropy_yface(i,j,k,0) + H_anisotropy_yface(i,j+1,k,0) );   
                plt(i,j,k,41) = 0.5 * ( H_anisotropy_zface(i,j,k,0) + H_anisotropy_zface(i,j,k+1,0) );  
 
                //Hy_anisotropy at xface, yface, zface
                plt(i,j,k,42) = 0.5 * ( H_anisotropy_xface(i,j,k,1) + H_anisotropy_xface(i+1,j,k,1) );   
                plt(i,j,k,43) = 0.5 * ( H_anisotropy_yface(i,j,k,1) + H_anisotropy_yface(i,j+1,k,1) );   
                plt(i,j,k,44) = 0.5 * ( H_anisotropy_zface(i,j,k,1) + H_anisotropy_zface(i,j,k+1,1) );  
 
                //Hz_anisotropy at xface, yface, zface
                plt(i,j,k,45) = 0.5 * ( H_anisotropy_xface(i,j,k,2) + H_anisotropy_xface(i+1,j,k,2) );   
                plt(i,j,k,46) = 0.5 * ( H_anisotropy_yface(i,j,k,2) + H_anisotropy_yface(i,j+1,k,2) );   
                plt(i,j,k,47) = 0.5 * ( H_anisotropy_zface(i,j,k,2) + H_anisotropy_zface(i,j,k+1,2) );  
 
            });
        }
} 
