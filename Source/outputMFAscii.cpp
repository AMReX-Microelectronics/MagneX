#include <AMReX_MultiFab.H> 
#include "MagneX.H"

using namespace amrex;


void outputMFAscii(const Array<MultiFab, AMREX_SPACEDIM>& Mfield, std::string filename)
{
//     BL_PROFILE_VAR("outputMFAscii()",outputMFAscii);
//     std::string plotfilename = filename;
//     std::ofstream ofs(plotfilename, std::ofstream::out);
    
//     for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
//     // for (MFIter mfi(output); mfi.isValid(); ++mfi) {
//         // ofs<<std::setprecision(16)<< Mfield[2][mfi]<<std::endl;                                              
//     }
//     ofs.close();
}