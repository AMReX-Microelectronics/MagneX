#include "Diagnostics.H"

void mf_avg_fc_to_cc(MultiFab&  Plt,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Mfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
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
} 
