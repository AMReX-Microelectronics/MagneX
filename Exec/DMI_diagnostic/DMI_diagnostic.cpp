
#include <fstream>
#include <iostream>

#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

using namespace std;
using namespace amrex;

static
void
PrintUsage (const char* progName)
{
    Print() << std::endl
            << "Use this utility to extract a 1D profile along the x-axis of Mz and theta for a DMI skyrmion test." << std::endl << std::endl;
    Print() << "Usage:" << '\n';
    Print() << progName << " infile=inputFileName" << '\n' << '\n';

    exit(1);
}

int
main (int   argc,
      char* argv[])
{
    amrex::Initialize(argc,argv);

    if (argc == 1) {
        PrintUsage(argv[0]);
    }

    // plotfile name
    std::string iFile;

    // read in parameters from inputs file
    ParmParse pp;

    // read in plotfile name
    pp.query("infile", iFile);
    if (iFile.empty())
        amrex::Abort("You must specify `infile'");

    // for the Header
    std::string Header = iFile;
    Header += "/Header";

    // open header
    ifstream x;
    x.open(Header.c_str(), ios::in);

    // read in first line of header
    string str;
    x >> str;

    // read in number of components from header
    int ncomp;
    x >> ncomp;

    // read in variable names from header
    for (int n=0; n<ncomp; ++n) {
        x >> str;
    }

    // read in dimensionality from header
    int dim;
    x >> dim;

    if (dim != AMREX_SPACEDIM) {
        Print() << "\nError: you are using a " << AMREX_SPACEDIM << "D build to open a "
                << dim << "D plotfile\n\n";
        Abort();
    }

    int lev = 0;

    do {

        if (lev > 9) {
            Abort("Utility only works for 10 levels of refinement or less");
        }

        // storage for the MultiFab
        MultiFab mf;

        std::string iFile_lev = iFile;

        std::string levX  = "/Level_"+to_string(lev)+"/Cell";
        std::string levXX = "/Level_0"+to_string(lev)+"/Cell";

        // now read in the plotfile data
        // check to see whether the user pointed to the plotfile base directory
        // or the data itself
        if (amrex::FileExists(iFile+levX+"_H")) {
            iFile_lev += levX;
        } else if (amrex::FileExists(iFile+levXX+"_H")) {
            iFile_lev += levXX;
        } else {
            break; // terminate while loop
        }

        // read in plotfile to MultiFab
        VisMF::Read(mf, iFile_lev);

        if (lev == 0) {
            ncomp = mf.nComp();
            Print() << "Number of components in the plotfile = " << ncomp << std::endl;
            Print() << "Nodality of plotfile = " << mf.ixType().toIntVect() << std::endl;
        }

        // get boxArray to compute number of grid points at the level
        BoxArray ba = mf.boxArray();
        Print() << "Number of grid points at level " << lev << " = " << ba.numPts() << std::endl;

        for ( MFIter mfi(mf,false); mfi.isValid(); ++mfi ) {

            const Box& bx = mfi.validbox();
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);

            const Array4<Real>& mfdata = mf.array(mfi);

            Real offset = 0.;
            
            int k = (hi.z+1)/2;
            int j = (hi.y+1)/2;
            for (auto i = (hi.x+1)/2; i <= hi.x; ++i) {
                std::cout << i << " " << mfdata(i,j,k,3)/1.1e6 << " " << mfdata(i,j,k,16)+offset << "\n";
                // 2pi cyclic fix
                if (i<hi.x) {
                    if (offset==0. && mfdata(i,j,k,16)>5. && mfdata(i+1,j,k,16)<1.) {
                        offset = 2.*M_PI;
                    }
                }
            }
           
        } // end MFIter

        // proceed to next level of refinement
        ++lev;

    } while(true);

}
