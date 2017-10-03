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
template<typename GroupType>
class Kernel
{
public:
    virtual ~Kernel(){}

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< How compatible is this kernel generator with this synapse group?
    //!< Ascending values indicate compatibility and negative numbers indicate incompatible
    virtual int getCompatibility(const GroupType &sg) const = 0;
};

//----------------------------------------------------------------------------
// KernelGPU
//----------------------------------------------------------------------------
template<typename GroupType>
class KernelGPU : public Kernel<GroupType>
{
public:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef typename std::map<std::string, GroupType>::const_iterator GroupIter;
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
    virtual unsigned int getMaxNumThreads(const GroupType &sg) const = 0;

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    //!< Determine how many threads this synapse group
    //!< requires, taking into account block size etc
    unsigned int getPaddedSize(const GroupType &sg) const
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

//------------------------------------------------------------------------
// GPUStaticGrid
//------------------------------------------------------------------------
template<typename GroupType, typename... Arguments>
class GPUStaticGrid
{
public:
    //------------------------------------------------------------------------
    // IWriter
    //------------------------------------------------------------------------
    class IWriter
    {
    public:
        virtual void generateGlobals(CodeStream &os, const std::string &ftype, Arguments... arguments) const = 0;
        virtual void generateGroup(CodeStream &os, const GroupType &group, const std::string &ftype, Arguments... arguments) const = 0;
    };

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    static void writeKernelCall(CodeStream &os, const KernelGPU<GroupType> *kernel, bool timingEnabled)
    {
        const unsigned int gridSizeBlocks = kernel->getMaxGridSizeBlocks();
        os << "// " << kernel->getKernelName() << " grid size = " << gridSizeBlocks << std::endl;
        os << CodeStream::OB(1131) << std::endl;

        // Declare threads and grid
        os << "dim3 threads(" << kernel->getBlockSize() << ", 1);" << std::endl;
        os << "dim3 grid(" << gridSizeBlocks << ", 1);" << std::endl;

        // Write code to record kernel start time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStart);" << std::endl;
        }

        // Write call to kernel, passing in any extra global parameters and time
        os << kernel->getKernelName() << " <<<grid, threads>>>(";
        for(const auto &p : kernel->getExtraGlobalParameters()) {
            os << p.first << ", ";
        }
        os << "t);" << std::endl;

        // Write code to record kernel stop time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStop);" << std::endl;
        }
        os << CodeStream::CB(1131) << std::endl;
    }

    static void generateKernel(CodeStream &os, const KernelGPU<GroupType> *kernel, const IWriter *writer,
                               const std::string &ftype, Arguments... arguments)
    {
        os << "extern \"C\" __global__ void " << kernel->getKernelName() << "(";
        for (const auto &p : kernel->getExtraGlobalParameters()) {
            os << p.second << " " << p.first << ", ";
        }
        os << ftype << " t)" << std::endl; // end of synapse kernel header
        os << CodeStream::OB(75);

        // Generate globals
        writer->generateGlobals(os, ftype, arguments...);

        // Loop through the groups
        for(const auto &g : kernel->getGrid()) {
            os << "// group " << std::get<0>(g)->first << std::endl;
            os << "if ((id >= " << std::get<1>(g) << ") && (id < " << std::get<2>(g) << "))" << CodeStream::OB(77);
            os << "const unsigned int lid = id - " << std::get<1>(g) << ";" << std::endl;

            // Generate group code
            writer->generateGroup(os, std::get<0>(g)->second, ftype, arguments...);

            os << CodeStream::CB(77);
            os << std::endl;
        }

        os << CodeStream::CB(75);
    }
};
