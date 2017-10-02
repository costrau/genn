#include "synapticEventKernel/base.h"

// GeNN includes
#include "codeStream.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::Base
//------------------------------------------------------------------------
void SynapticEventKernel::Base::addSynapseGroup(SynapseGroupIter sg)
{
    // Add synapse group iter to grid
    m_Grid.push_back(std::make_tuple(sg, 0, 0));

    // Add extra global synapse parameters
    sg->second.addExtraGlobalSynapseParams(m_ExtraGlobalParameters);
}
//------------------------------------------------------------------------
void SynapticEventKernel::Base::getNumThreads(std::vector<unsigned int> &groupSizes) const
{
    // Reserve group sizes
    groupSizes.reserve(m_Grid.size());

    // Loop through grid and add number of threads
    for(const auto &s : m_Grid) {
        groupSizes.push_back(getNumThreads(std::get<0>(s)->second));
    }
}
//------------------------------------------------------------------------
void SynapticEventKernel::Base::setBlockSize(unsigned int blockSize)
{
    // Set block size
    m_BlockSize = blockSize;

    // Loop through synapse groups in grid
    unsigned int idStart = 0;
    for(auto &s : m_Grid) {
        // Update starting index
        std::get<1>(s) = idStart;

        // Add padded size of this synapse group to id
        idStart += getPaddedSize(std::get<0>(s)->second);

        // Update end index of this synapse in grid
        std::get<2>(s) = idStart;
    }
}
//------------------------------------------------------------------------
unsigned int SynapticEventKernel::Base::getGridSize() const
{
    if(m_Grid.empty()) {
        return 0;
    }
    else {
        return std::get<2>(m_Grid.back());
    }
}
//------------------------------------------------------------------------
void SynapticEventKernel::Base::writeKernelCall(CodeStream &os, bool timingEnabled) const
{
    const unsigned int gridSize = getGridSize();
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
    os << CodeStream::CB(1131) << std::endl;
}
//----------------------------------------------------------------------------
unsigned int SynapticEventKernel::Base::getPaddedSize(const SynapseGroup &sg) const
{
    return (unsigned int)(ceil((double)getNumThreads(sg) / (double)getBlockSize()) * (double)getBlockSize());
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