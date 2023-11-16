#include <AMReX_MultiFab.H>

using namespace amrex;

long CountMagneticCells(MultiFab& Ms);

Real SumNormalizedM(MultiFab& Ms,
                    MultiFab& Mfield);
