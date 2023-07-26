#include "AMReX_PlotFileUtil.H"
#include "AMReX_PlotFileDataImpl.H"

#include <sys/stat.h>

#include "Checkpoint.H"

namespace {
    void GotoNextLine (std::istream& is)
    {
        constexpr std::streamsize bl_ignore_max { 100000 };
        is.ignore(bl_ignore_max, '\n');
    }
}

void WriteCheckPoint(int step,
                     const amrex::Real time,
                     Array< MultiFab, AMREX_SPACEDIM>& Mfield,
                     Array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
		     Array< MultiFab, AMREX_SPACEDIM>& H_demagfield)
{
    // timer for profiling
    BL_PROFILE_VAR("WriteCheckPoint()",WriteCheckPoint);

    // checkpoint file name, e.g., chk0000010
    const std::string& checkpointname = amrex::Concatenate("chk",step,8);

    amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

    BoxArray ba = H_demagfield[0].boxArray();

    // single level problem
    int nlevels = 1;

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);
    
    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    // write Header file
    if (ParallelDescriptor::IOProcessor()) {

        std::ofstream HeaderFile;
        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        std::string HeaderFileName(checkpointname + "/Header");
        HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                        std::ofstream::trunc |
                        std::ofstream::binary);

        if( !HeaderFile.good()) {
            amrex::FileOpenFailed(HeaderFileName);
        }

        HeaderFile.precision(17);

        // write out title line
        HeaderFile << "Checkpoint file for MagneX\n";

        // write out time
        HeaderFile << time << "\n";
        
        // write the BoxArray (fluid)
        ba.writeOn(HeaderFile);
        HeaderFile << '\n';
    }

    // write the MultiFab data to, e.g., chk00010/Level_0/
    VisMF::Write(Mfield[0],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldx"));
    VisMF::Write(Mfield[1],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldy"));
    VisMF::Write(Mfield[2],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldz"));
    VisMF::Write(H_biasfield[0],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_biasfieldx"));
    VisMF::Write(H_biasfield[1],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_biasfieldy"));
    VisMF::Write(H_biasfield[2],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_biasfieldz"));
    VisMF::Write(H_demagfield[0],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_demagfieldx"));
    VisMF::Write(H_demagfield[1],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_demagfieldy"));
    VisMF::Write(H_demagfield[2],
                 amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_demagfieldz"));
}

void ReadCheckPoint(int& restart,
		    amrex::Real& time,
		    Array< MultiFab, AMREX_SPACEDIM>& Mfield,
		    Array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
		    Array< MultiFab, AMREX_SPACEDIM>& H_demagfield,
		    BoxArray& ba,
		    DistributionMapping& dm)
{
    // timer for profiling
    BL_PROFILE_VAR("ReadCheckPoint()",ReadCheckPoint);

    // checkpoint file name, e.g., chk0000010
    const std::string& checkpointname = amrex::Concatenate("chk",restart,8);

    amrex::Print() << "Restart from checkpoint " << checkpointname << "\n";

    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

    std::string line, word;

    // Header
    {
        std::string File(checkpointname + "/Header");
        Vector<char> fileCharPtr;
        ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
        std::string fileCharPtrString(fileCharPtr.dataPtr());
        std::istringstream is(fileCharPtrString, std::istringstream::in);

        // read in title line
        std::getline(is, line);

        // read in time
        is >> time;
        GotoNextLine(is);

        // read in BoxArray from Header
        ba.readFrom(is);
        GotoNextLine(is);

        // create a distribution mapping
        dm.define(ba, ParallelDescriptor::NProcs());

	int Nghost = 2;
	
//	AMREX_D_TERM(Mfield[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, Nghost);,
//		     Mfield[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, Nghost);,
//		     Mfield[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, Nghost);)
//	
//	AMREX_D_TERM(H_biasfield[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 3, 0);,
//		     H_biasfield[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 3, 0);,
//		     H_biasfield[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 3, 0););

	for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
	  Mfield[dir].define(ba, dm, 1, Nghost);
	  H_demagfield[dir].define(ba, dm, 1, 1);
	  H_biasfield[dir].define(ba, dm, 1, 0);
	}
    }

    // read in the MultiFab data
    VisMF::Read(Mfield[0],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldx"));
    VisMF::Read(Mfield[1],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldy"));
    VisMF::Read(Mfield[2],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldz"));
    VisMF::Read(H_biasfield[0],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_biasfieldx"));
    VisMF::Read(H_biasfield[1],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_biasfieldy"));
    VisMF::Read(H_biasfield[2],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_biasfieldz"));
    VisMF::Read(H_demagfield[0],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_demagfieldx"));
    VisMF::Read(H_demagfield[1],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_demagfieldy"));
    VisMF::Read(H_demagfield[2],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "H_demagfieldz"));
}


