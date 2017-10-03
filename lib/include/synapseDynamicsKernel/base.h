#pragma once

// Standard C++ includes
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "kernel.h"
#include "neuronGroup.h"
#include "synapseGroup.h"

//----------------------------------------------------------------------------
// SynapseDynamicsKernel::BaseGPU
//----------------------------------------------------------------------------
//!< Base class for synapse dynamics kernels which use GPU
namespace SynapseDynamicsKernel
{
class BaseGPU : public KernelGPU<SynapseGroup>
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< Generate a kernel for simulating the specified subset
    //!< of synapse groups and write it to the CodeStream
    virtual void generateKernel(CodeStream &os, const std::string &ftype) const = 0;
};


//------------------------------------------------------------------------
// SynapticEventKernel::BaseStaticGrid
//------------------------------------------------------------------------
typedef GPUStaticGrid<SynapseGroup> StaticGrid;
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
        sg->second.addExtraGlobalSynapseDynamicsParams(m_ExtraGlobalParameters);
    }

    //!< Generate a kernel for simulating the specified subset
    //!< of synapse groups and write it to the CodeStream
    virtual void generateKernel(CodeStream &os, const std::string &ftype) const override
    {
        StaticGrid::generateKernel(os, this, this, ftype);
    }

    virtual void writeKernelCall(CodeStream &os, bool timingEnabled) const override
    {
        StaticGrid::writeKernelCall(os, this, timingEnabled);
    }

};

}   // namespace SynapseDynamicsKernel