#include "MagneX.H"

#include "AMReX_PlotFileUtil.H"

namespace {
    void GotoNextLine (std::istream& is)
    {
        constexpr std::streamsize bl_ignore_max { 100000 };
        is.ignore(bl_ignore_max, '\n');
    }
}

void WriteCheckPoint(int step,
                     const amrex::Real time,
                     Array< MultiFab, AMREX_SPACEDIM>& Mfield)
{
    // timer for profiling
    BL_PROFILE_VAR("WriteCheckPoint()",WriteCheckPoint);

    // checkpoint file name, e.g., chk0000010
    const std::string& checkpointname = amrex::Concatenate("chk",step,8);

    amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

    BoxArray ba = Mfield[0].boxArray();

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
}

void ReadCheckPoint(int& restart,
		    amrex::Real& time,
		    Array< MultiFab, AMREX_SPACEDIM>& Mfield,
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

        int Nghost = 1;

        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            Mfield[dir].define(ba, dm, 1, Nghost);
        }
    }

    // read in the MultiFab data
    VisMF::Read(Mfield[0],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldx"));
    VisMF::Read(Mfield[1],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldy"));
    VisMF::Read(Mfield[2],
                amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Mfieldz"));
}


