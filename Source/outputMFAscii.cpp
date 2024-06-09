#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>

using namespace amrex;


void outputMFAscii(const Array<MultiFab, AMREX_SPACEDIM>& Mfield, std::string filename)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        VisMF::Write(Mfield[idim], filename+std::to_string(idim));
    }                        
}