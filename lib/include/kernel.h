#pragma once

// Standard C++ includes
#include <map>
#include <string>
#include <tuple>
#include <vector>

// Standard C includes
#include <cmath>

// GeNN includes
#include "codeStream.h"

//----------------------------------------------------------------------------
// Kernel
//----------------------------------------------------------------------------
template<typename G>
class Kernel
{
public:
    virtual ~Kernel(){}

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef G GroupType;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< How compatible is this kernel generator with this synapse group?
    //!< Ascending values indicate compatibility and negative numbers indicate incompatible
    virtual int getCompatibility(const G &sg) const = 0;
};

//----------------------------------------------------------------------------
// KernelGPU
//----------------------------------------------------------------------------
template<typename G>
class KernelGPU : public Kernel<G>
{
public:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef typename std::map<std::string, G>::const_iterator GroupIter;
    typedef std::tuple<GroupIter, unsigned int, unsigned int> GridEntry;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< Get the name of the kernel (used to call it from runner)
    virtual std::string getKernelName() const = 0;

    //!< Write code to define grid and call kernel
    virtual void writeKernelCall(CodeStream &os, bool timingEnabled) const = 0;

    //!< Add a synapse group to be generated with this event kernel generator
    virtual void addGroup(GroupIter sg)
    {
        // Add synapse group iter to grid
        m_Grid.push_back(std::make_tuple(sg, 0, 0));
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //!< Is this kernel in use
    bool isUsed() const
    {
        return !m_Grid.empty();
    }

    //!< Gets number of threads each synapse group this kernel is used to simulate will require
    void getMaxNumThreads(std::vector<unsigned int> &groupMaxSizes) const
    {
         // Reserve group sizes
        groupMaxSizes.reserve(m_Grid.size());

        // Loop through grid and add number of threads
        for(const auto &s : m_Grid) {
            groupMaxSizes.push_back(getMaxNumThreads(std::get<0>(s)->second));
        }
    }

    //!< Set block size and re-evaluate grid based on new block size
    void setBlockSize(unsigned int blockSize)
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

    //!< Get  size of each thread block
    unsigned int getBlockSize() const
    {
        return m_BlockSize;
    }

    //!< Get total size of grid in terms of threads
    unsigned int getMaxGridSizeThreads() const
    {
        if(m_Grid.empty()) {
            return 0;
        }
        else {
            return std::get<2>(m_Grid.back());
        }
    }

    //!< Get total size of grid in terms of blocks
    unsigned int getMaxGridSizeBlocks() const
    {
        const unsigned int maxGridSizeThreads = getMaxGridSizeThreads();
        if(maxGridSizeThreads == 0) {
            return 0;
        }
        else {
            return ceil((float)maxGridSizeThreads / getBlockSize());
        }
    }

    const std::map<std::string, std::string> &getExtraGlobalParameters() const
    {
        return m_ExtraGlobalParameters;
    }

    //!< Gets grid to simulate in this kernel
    const std::vector<GridEntry> &getGrid() const
    {
        return m_Grid;
    }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< Determine how many threads this synapse group
    //!< requires, not taking into account block size etc
    virtual unsigned int getMaxNumThreads(const G &sg) const = 0;

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    //!< Determine how many threads this synapse group
    //!< requires, taking into account block size etc
    unsigned int getPaddedSize(const G &sg) const
    {
        return (unsigned int)(ceil((double)getMaxNumThreads(sg) / (double)getBlockSize()) * (double)getBlockSize());
    }

    // **HACK**
    std::map<std::string, std::string> m_ExtraGlobalParameters;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //!< Extra gloval parameters to pass to this kernel
    //std::map<std::string, std::string> m_ExtraGlobalParameters;

    //!< Synapse groups and the ids at which they end
    std::vector<GridEntry> m_Grid;

    //!< How large is the block size used by this kernel
    unsigned int m_BlockSize;
};
