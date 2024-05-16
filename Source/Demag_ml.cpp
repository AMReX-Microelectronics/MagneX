#include "MagneX.H"
#include <torch/script.h>
using namespace amrex;

void CalculateH_demag_ML(Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                        torch::jit::script::Module& x_norm_module,
                        torch::jit::script::Module& ml_module,
                        torch::jit::script::Module& y_norm_module,
                        Array<MultiFab, AMREX_SPACEDIM>& H_demagfield)

{
    // timer for profiling
    BL_PROFILE_VAR("CalculateH_demag_ML()",CalculateH_demag_ML);
    auto dtype0 = torch::kFloat64;
    std::string plotfilename = std::to_string(ParallelDescriptor::MyProc()) + "_sample_output";
    std::ofstream ofs(plotfilename, std::ofstream::out);
    // amrex::Print()<<"\n"<<"AMREX_SPACEDIM " << AMREX_SPACEDIM << "\n";
    // for (MFIter mfi(output); mfi.isValid(); ++mfi) {
    //     ofs<<std::setprecision(16)<< (output[mfi])<<std::endl;                                              
    // }
    // ofs.close();

    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    
                    const Box& bx = mfi.validbox();
                    const Array4<Real>& Mx = Mfield[0].array(mfi);
                    const Array4<Real>& My = Mfield[1].array(mfi); 
                    const Array4<Real>& Mz = Mfield[2].array(mfi); 
                    // ofs<<std::setprecision(16)<< Mfield[0][mfi] <<std::endl;   
                    ofs<<std::setprecision(16)<< Mfield[2][mfi] <<std::endl;   

                    const Array4<Real>& Hx_demag = H_demagfield[0].array(mfi);   
                    const Array4<Real>& Hy_demag = H_demagfield[1].array(mfi);   
                    const Array4<Real>& Hz_demag = H_demagfield[2].array(mfi);   
                    // ofs<<std::setprecision(16)<< H_demagfield[1][mfi] <<std::endl;


                    const IntVect bx_lo = bx.smallEnd();
                    const IntVect nbox = bx.size();
                    int ncell = AMREX_SPACEDIM == 2 ?
                        nbox[0] * nbox[1] : nbox[0] * nbox[1] * nbox[2];
                    amrex::Gpu::ManagedVector<Real> aux_Mx(ncell*1);
                    Real* AMREX_RESTRICT auxPtr_Mx = aux_Mx.dataPtr();
                    amrex::Gpu::streamSynchronize();
                    amrex::Gpu::ManagedVector<Real> aux_My(ncell*1);
                    Real* AMREX_RESTRICT auxPtr_My = aux_My.dataPtr();
                    amrex::Gpu::streamSynchronize();
                    amrex::Gpu::ManagedVector<Real> aux_Mz(ncell*1);
                    Real* AMREX_RESTRICT auxPtr_Mz = aux_Mz.dataPtr();

                    // amrex::Print()<<"\n"<<"total_cells: " << ncell << "\n";            
                    amrex::Gpu::streamSynchronize();
                    // copy input multifab to torch tensor
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        int ii = i - bx_lo[0];
                        int jj = j - bx_lo[1];
                        int index = jj*nbox[0] + ii;
                    #if AMREX_SPACEDIM == 3
                        int kk = k - bx_lo[2];
                        index += kk*nbox[0]*nbox[1];
                    #endif
                        // array order is row-based [index][comp]
                        auxPtr_Mx[index] = Mx(i, j, k);
                        auxPtr_My[index] = My(i, j, k);
                        auxPtr_Mz[index] = Mz(i, j, k);

                        // printf("Mx value is:%g", Mx(i,j,k));
                        // printf("i=%d, j=%d, k=%d, ii=%d, jj=%d, bx_low_0=%d, bx_low_1=%d,nbox_0=%d, idx=%d, Mx value is:%g\n", i, j, k, ii, jj, bx_lo[0], bx_lo[1], bx_lo[2], index, Mx(i,j,k));
                        // printf("idx=%d, auxPtr_Mx value is:%g\n", index, auxPtr_Mx[index + 1]);
                        // amrex::Print() << "i,j,k" << i << " " << j << " " << k << " index " << index << " Mx " << Mx(i,j,k,n) << " auxPtr_Mx " << auxPtr_Mx[index] << std::endl;
                    }); 
                    amrex::Gpu::streamSynchronize();

                    // printf("idx=%d, auxPtr_Mx value is:%g\n", 0, auxPtr_Mx[0 + 1]);
                    // amrex::Print() << "auxPtr_Mx_11=" << auxPtr_Mx[1] << "\n";
                    // amrex::Print() << "auxPtr_Mx_22=" << auxPtr_Mx[2] << "\n";
                    // amrex::Print() << "auxPtr_Mx_33=" << auxPtr_Mx[3] << "\n";
                    // amrex::Print() << "auxPtr_Mx_44=" << auxPtr_Mx[4] << "\n";

                    // for (int idx = 0; idx < ncell; ++idx) {
                    //     printf("idx=%d, auxPtr_Mx current value is:%g\n", idx, auxPtr_Mx[idx]);

                    // }


                    at::Tensor inputs_torch_Mx = torch::from_blob(auxPtr_Mx, {ncell, 1}, torch::kFloat64);
                    at::Tensor inputs_torch_My = torch::from_blob(auxPtr_My, {ncell, 1}, torch::kFloat64);
                    at::Tensor inputs_torch_Mz = torch::from_blob(auxPtr_Mz, {ncell, 1}, torch::kFloat64);

                    // std::cout << "Tensor check:" << std::endl;
                    // for (int i = 0; i < ncell; i++) {
                    //     std::cout << "Tensor[" << i << "] = " << inputs_torch_Mx[i].item<float>() << std::endl;
                    // }   

                    // amrex::Print()<<"\n"<<"inputs_torch_Mx: " << inputs_torch_Mx << "\n";    
                    // amrex::Print() << "auxPtr_Mx_1=" << auxPtr_Mx[1] << "\n";
                    // amrex::Print() << "auxPtr_Mx_2=" << auxPtr_Mx[2] << "\n";
                    // amrex::Print() << "auxPtr_Mx_3=" << auxPtr_Mx[3] << "\n";
                    // amrex::Print() << "auxPtr_Mx_4=" << auxPtr_Mx[4] << "\n";

                    // at::Tensor inputs_torch_Mx2 = torch::from_blob(auxPtr_Mx, {ncell, 1}, tensoropt);
                    // at::Tensor inputs_torch_My2 = torch::from_blob(auxPtr_My, {ncell, 1}, tensoropt);
                    // at::Tensor inputs_torch_Mz2 = torch::from_blob(auxPtr_Mz, {ncell, 1}, tensoropt);
                    // amrex::Print()<<"\n"<<"inputs_torch_Mx2: " << inputs_torch_Mx << "\n";  
                    // printf("idx=1, auxPtr_Mx value is:%g\n", auxPtr_Mx[1]);
                    // Reshape each tensor to {166, 42}
                    at::Tensor reshaped_Mx = inputs_torch_Mx.reshape({166, 42});
                    at::Tensor reshaped_My = inputs_torch_My.reshape({166, 42});
                    at::Tensor reshaped_Mz = inputs_torch_Mz.reshape({166, 42});

                    // amrex::Print()<<"\n"<<"reshaped_Mx: " << reshaped_Mx << "\n";    
                    // amrex::Print()<<"\n"<<"reshaped_My: " << reshaped_My << "\n";    
                    amrex::Print()<<"\n"<<"reshaped_Mz: " << reshaped_Mz << "\n";    



                    // Concatenate all reshaped tensors along a new dimension
                    at::Tensor final_tensor_M = torch::stack({reshaped_Mx, reshaped_My, reshaped_Mz}, 0);


                    // amrex::Print()<<"\n"<<"input torch M: " << final_tensor_M << "\n";

                    final_tensor_M = final_tensor_M.to(torch::kCUDA);
                    final_tensor_M = final_tensor_M.to(torch::kFloat32);
                    // amrex::Print()<<"\n"<<"input torch M: " << final_tensor_M << "\n";
                    
                    final_tensor_M = final_tensor_M.unsqueeze(0);
                    final_tensor_M.to(torch::kCUDA);
                    at::Tensor norm_torch = x_norm_module.get_method("encode")({final_tensor_M}).toTensor();
                    // amrex::Print()<<"\n"<<"output torch M: " << norm_torch << "\n";
                    // norm_torch = norm_torch.permute({0, 2, 3, 1});
                    at::Tensor outputs_torch = ml_module.forward({norm_torch}).toTensor();
                    // outputs_torch = outputs_torch.permute({0, 3, 1, 2});
                    at::Tensor denorm_torch = y_norm_module.get_method("decode")({outputs_torch}).toTensor();
                    denorm_torch = denorm_torch.to(torch::kFloat64);
                    // amrex::Print()<<"\n"<<"output torch M denorm: " << denorm_torch << "\n";


                    //construct output to H_demag field
                    at::Tensor denorm_torch_Hx = denorm_torch.select(0, 0).select(0, 0);
                    at::Tensor denorm_torch_Hy = denorm_torch.select(0, 0).select(0, 1);
                    at::Tensor denorm_torch_Hz = denorm_torch.select(0, 0).select(0, 2);
                    // amrex::Print()<<"\n"<<"output torch Hx denorm: " << denorm_torch_Hx << "\n";
                    // amrex::Print()<<"\n"<<"output torch Hy denorm: " << denorm_torch_Hy << "\n";
                    // amrex::Print()<<"\n"<<"output torch Hz denorm: " << denorm_torch_Hz << "\n";

                    // at::Tensor reshaped_Mx = inputs_torch_Mx.reshape({166, 42});
                    // denorm_torch_Hx = denorm_torch_Hx.reshape({ncell, 1});
                    // denorm_torch_Hy = denorm_torch_Hy.reshape({ncell, 1});
                    // denorm_torch_Hz = denorm_torch_Hz.reshape({ncell, 1});
                    denorm_torch_Hx = denorm_torch_Hx.transpose(0, 1).flatten();
                    denorm_torch_Hy = denorm_torch_Hy.transpose(0, 1).flatten();
                    denorm_torch_Hz = denorm_torch_Hz.transpose(0, 1).flatten();

                    // amrex::Print()<<"\n"<<"output torch Hx denorm reshape: " << denorm_torch_Hx << "\n";

                    #ifdef AMREX_USE_CUDA
                        auto denorm_torch_Hx_acc = denorm_torch_Hx.packed_accessor64<Real,1>();
                        auto denorm_torch_Hy_acc = denorm_torch_Hy.packed_accessor64<Real,1>();
                        auto denorm_torch_Hz_acc = denorm_torch_Hz.packed_accessor64<Real,1>();

                    #else
                        // auto reshaped_Mx_test_acc = reshaped_Mx_test.accessor<Real,2>();
                    #endif

                    // amrex::Print()<<"\n"<<"denorm_torch_Hx_acc: " << denorm_torch_Hx_acc[0][0] << "\n";

                    // copy tensor to output multifab
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        int ii = i - bx_lo[0];
                        int jj = j - bx_lo[1];
                        int index = jj*nbox[0] + ii;
                        #if AMREX_SPACEDIM == 3
                            int kk = k - bx_lo[2];
                            index += kk*nbox[0]*nbox[1];
                        #endif
                        Hx_demag(i, j, k) = denorm_torch_Hx_acc[index];
                        // printf("i=%d, j=%d, k=%d, ii=%d, jj=%d, bx_low_0=%d, bx_low_1=%d,nbox_0=%d, idx=%d, Hx_demag value is:%g\n", i, j, k, ii, jj, bx_lo[0], bx_lo[1], bx_lo[2], index, denorm_torch_Hx_acc[index]);
                        Hy_demag(i, j, k) = denorm_torch_Hy_acc[index];
                        Hz_demag(i, j, k) = denorm_torch_Hz_acc[index];
                        // ofs<<std::setprecision(16)<< H_demagfield[0][mfi] <<std::endl;
                        // ofs<<std::setprecision(16)<< H_demagfield[0][mfi] <<std::endl;      
                        
                    });
                    amrex::Gpu::streamSynchronize();
                    // ofs<<std::setprecision(16)<< H_demagfield[2][mfi] <<std::endl;


    }
    // ofs.close(); 
}

