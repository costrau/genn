#include "synapticEventKernel/base.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::Base
//------------------------------------------------------------------------
void SynapticEventKernel::Base::addSynapseGroup(SynapseGroupIter sg)
{
    // Add synapse group iter to grid
    m_Grid.push_back(std::make_pair(sg, 0));

    // Add extra global synapse parameters
    sg->first.addExtraGlobalSynapseParams(m_ExtraGlobalParameters);
}
//------------------------------------------------------------------------
void SynapticEventKernel::Base::setBlockSize(unsigned int blockSize)
{
    // Set block size
    m_BlockSize = blocKSize;

    // Loop through synapse groups in grid
    unsigned int idStart = 0;
    for(auto &s : m_Grid) {
        // Add padded size of this synapse group to id
        idStart += getPaddedSize(*s.first);

        // Update end index of this synapse in grid
        s.second = idStart;
    }
}
//------------------------------------------------------------------------
void SynapticEventKernel::Base::writeKernelCall(CodeStream &os, bool timingEnabled) const
{
    // Grid size is the last ID of the last synapse group in grid
    const unsigned int gridSize = m_Grid.back().second;

    os << "// " << getKernelName() << " grid size = " << gridSize << std::endl;
    os << CodeStream::OB(1131) << std::endl;

    // Declare threads and grid
    os << "dim3 sThreads(" << getBlockSize() << ", 1);" << std::endl;
    os << "dim3 sGrid(" << gridSize << ", 1);" << std::endl;

    // Write code to record kernel start time
    if(timingEnabled) {
        os << "cudaEventRecord(synapseStart);" << std::endl;
    }

    // Write call to kernel, passing in any extra global parameters and time
    os << getKernelName() << " <<<sGrid, sThreads>>>(";
    for(const auto &p : m_ExtraGlobalParameters) {
        os << p.first << ", ";
    }
    os << "t);" << std::endl;

    // Write code to record kernel stop time
    if(timingEnabled) {
        os << "cudaEventRecord(synapseStop);" << std::endl;
    }
    os << CodeStream::CB(1131) << std::end;
}
//------------------------------------------------------------------------
void SynapticEventKernel::Base::writeKernelDeclaration(CodeStream &os, const std::string &ftype) const
{
    os << "extern \"C\" __global__ void " << getKernelName() << "(";
    for (const auto &p : m_ExtraGlobalParameters) {
        os << p.second << " " << p.first << ", ";
    }
    os << ftype << " t)" << std::endl; // end of synapse kernel header
}