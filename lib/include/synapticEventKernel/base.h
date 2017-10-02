#pragma once

// Standard C++ includes
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "synapseGroup.h"

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

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef std::map<std::string, SynapseGroup>::const_iterator SynapseGroupIter;
    typedef std::tuple<SynapseGroupIter, unsigned int, unsigned int> GridEntry;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< How compatible is this kernel generator with this synapse group?
    //!< Ascending values indicate compatibility and negative numbers indicate incompatible
    virtual int getCompatibility(const SynapseGroup &sg) const = 0;

    //!< Generate a kernel for simulating the specified subset
    //!< of synapse groups and write it to the CodeStream
    virtual void generateKernel(CodeStream &os, bool isResetKernel,
                                unsigned int totalBlocks,
                                const std::map<std::string, NeuronGroup> &ngs,
                                const std::string &ftype) const = 0;

    //!< Get the name of the kernel (used to call it from runner)
    virtual std::string getKernelName() const = 0;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //!< Add a synapse group to be generated with this event kernel generator
    void addSynapseGroup(SynapseGroupIter sg);

    //!< Gets number of threads each synapse group this kernel is used to simulate will require
    void getNumThreads(std::vector<unsigned int> &groupSizes) const;

    //!< Is this kernel in use
    bool isUsed() const{ return !m_Grid.empty(); }

    // **TODO** GPU only
    //!< Write code to define grid and call kernel
    void writeKernelCall(CodeStream &os, bool timingEnabled) const;

    //!< Set block size and re-evaluate grid based on new block size
    void setBlockSize(unsigned int blockSize);

    //!< Get  size of each thread block
    unsigned int getBlockSize() const{ return m_BlockSize; }

    //!< Get total size of grid in terms of threads
    unsigned int getGridSizeThreads() const;

    //!< Get total size of grid in terms of blocks
    unsigned int getGridSizeBlocks() const;

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< Determine how many threads this synapse group
    //!< requires, not taking into account block size etc
    virtual unsigned int getNumThreads(const SynapseGroup &sg) const = 0;

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    //!< Write kernel function declaration to code stream
    void writeKernelDeclaration(CodeStream &os, const std::string &ftype) const;

    //!< Determine how many threads this synapse group
    //!< requires, taking into account block size etc
    unsigned int getPaddedSize(const SynapseGroup &sg) const;

    //!< Gets grid to simulate in this kernel
    const std::vector<GridEntry> &getGrid() const{ return m_Grid; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //!< Extra gloval parameters to pass to this kernel
    std::map<std::string, std::string> m_ExtraGlobalParameters;

    //!< Synapse groups and the ids at which they end
    std::vector<GridEntry> m_Grid;

    //!< How large is the block size used by this kernel
    unsigned int m_BlockSize;
};
}   // namespace SynapticEventKernel