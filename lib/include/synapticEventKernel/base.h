#pragma once

// Standard C++ includes
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "grid.h"
#include "kernel.h"
#include "neuronGroup.h"
#include "synapseGroup.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::BaseGPU
//----------------------------------------------------------------------------
//!< Base class for synaptic event kernels which use GPU
namespace SynapticEventKernel
{
class BaseGPU : public KernelGPU<SynapseGroup>
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< Generate a kernel for simulating the specified subset
    //!< of synapse groups and write it to the CodeStream
    virtual void generateKernel(CodeStream &os, bool isResetKernel,
                                unsigned int totalBlocks,
                                const std::map<std::string, NeuronGroup> &ngs,
                                const std::string &ftype) const = 0;
};


//------------------------------------------------------------------------
// SynapticEventKernel::BaseStaticGrid
//------------------------------------------------------------------------
typedef GPUStaticGrid<SynapseGroup, bool, unsigned int, const std::map<std::string, NeuronGroup>&> StaticGrid;
//------------------------------------------------------------------------
//!< Base class for synaptic event kernels which use the GPU and have a static grid
class BaseStaticGrid : public BaseGPU, public StaticGrid::IWriter
{
public:
    //------------------------------------------------------------------------
    // KernelGPU virtuals
    //------------------------------------------------------------------------
    virtual void addGroup(GroupIter sg)
    {
        // Superclass
        KernelGPU<SynapseGroup>::addGroup(sg);

         // Add extra global synapse parameters
        sg->second.addExtraGlobalSynapseParams(m_ExtraGlobalParameters);
    }

    //!< Generate a kernel for simulating the specified subset
    //!< of synapse groups and write it to the CodeStream
    virtual void generateKernel(CodeStream &os, bool isResetKernel,
                                unsigned int totalBlocks,
                                const std::map<std::string, NeuronGroup> &ngs,
                                const std::string &ftype) const override
    {
        StaticGrid::generateKernel(os, this, this, ftype, isResetKernel, totalBlocks, ngs);
    }

    virtual void writeKernelCall(CodeStream &os, bool timingEnabled) const override
    {
        StaticGrid::writeKernelCall(os, this, timingEnabled);
    }

};

//------------------------------------------------------------------------
// SynapticEventKernel::BaseDynamicSpikeGrid
//------------------------------------------------------------------------
typedef GPUDynamicSpikeGrid<bool, unsigned int, const std::map<std::string, NeuronGroup>&> DynamicSpikeGrid;
//------------------------------------------------------------------------
//!< Base class for synaptic event kernels which use the GPU and have a dynamic spike grid
class BaseDynamicSpikeGrid : public BaseGPU, public DynamicSpikeGrid::IWriter
{
public:
    //------------------------------------------------------------------------
    // KernelGPU virtuals
    //------------------------------------------------------------------------
    virtual void addGroup(GroupIter sg)
    {
        // Superclass
        KernelGPU<SynapseGroup>::addGroup(sg);

         // Add extra global synapse parameters
        sg->second.addExtraGlobalSynapseParams(m_ExtraGlobalParameters);
    }

    //!< Generate a kernel for simulating the specified subset
    //!< of synapse groups and write it to the CodeStream
    virtual void generateKernel(CodeStream &os, bool isResetKernel,
                                unsigned int totalBlocks,
                                const std::map<std::string, NeuronGroup> &ngs,
                                const std::string &ftype) const override
    {
        DynamicSpikeGrid::generateKernel(os, this, this, ftype, isResetKernel, totalBlocks, ngs);
    }

    virtual void writeKernelCall(CodeStream &os, bool timingEnabled) const override
    {
        DynamicSpikeGrid::writeKernelCall(os, this, timingEnabled);
    }

};


}   // namespace SynapticEventKernel