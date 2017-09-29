#pragma once

// Standard C++ includes
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "../synapseGroup.h"

// Forward declarations
class CodeStream;

//----------------------------------------------------------------------------
// SynapticEventKernel::Base
//----------------------------------------------------------------------------
namespace SynapticEventKernel
{
class Base
{
public:
    Base() : m_BlockSize(0){}

    typedef std::map<std::string, SynapseGroup>::const_iterator SynapseGroupIter;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< How compatible is this kernel generator with this synapse group?
    //!< Ascending values indicate compatibility and negative numbers indicate incompatible
    int getCompatibility(const SynapseGroup &sg) const = 0;

    //!< Generate a kernel for simulating the specified subset
    //!< of synapse groups and write it to the CodeStream
    void generateKernel(CodeStream &os, bool isResetKernel) const = 0;

    //!< Get the name of the kernel (used to call it from runner)
    std::string getKernelName() const = 0;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //!< Add a synapse group to be generated with this event kernel generator
    void addSynapseGroup(const SynapseGroup &sg);

    //!< Re-evaluate grid based on new block size
    void updateGrid(unsigned int blockSize);

    //!< Write code to define grid and call kernel
    void writeKernelCall(CodeStream &os) const;

    //!< Is this kernel in use
    bool isUsed() const{ return !m_Grid.empty(); }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    //!< Write kernel function declaration to code stream
    void writeKernelDeclaration(CodeStream &os) const;

    //!< Gets current block size
    unsigned int getBlockSize() const{ return m_BlockSize; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //!< Extra gloval parameters to pass to this kernel
    std::map<std::string, std::string> m_ExtraGlobalParameters;

    //!< Synapse groups and the ids at which they end
    //!< **NOTE** first synapse group always starts at 0
    std::vector<std::tuple<SynapseGroupIter, unsigned int>> m_Grid;

    //!< How large is the block size used by this kernel
    unsigned int m_BlockSize;
};
}   // namespace SynapticEventKernel