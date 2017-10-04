#pragma once

// Standard C++ includes
#include <map>
#include <set>
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
    virtual void generateKernel(CodeStream &os, const std::string &ftype,
                                bool isResetKernel, const std::map<std::string, NeuronGroup> &ngs) const = 0;

    //!< Gets set of the names of any neuron groups whose
    //!< spike counts are required to calculate grid size
    virtual void getRequiredNeuronGroupSpikeCounts(std::set<const NeuronGroup*> &) const{}
};


//------------------------------------------------------------------------
// SynapticEventKernel::BaseStaticGrid
//------------------------------------------------------------------------
//!< Base class for synaptic event kernels which use the GPU and have a static grid
class BaseStaticGrid : public GPUStaticGrid<BaseGPU, const std::map<std::string, NeuronGroup>&>
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
};

//------------------------------------------------------------------------
// SynapticEventKernel::BaseDynamicSpikeGrid
//------------------------------------------------------------------------
//!< Base class for synaptic event kernels which use the GPU and have a dynamic spike grid
class BaseDynamicSpikeGrid : public GPUDynamicSpikeGrid<BaseGPU, const std::map<std::string, NeuronGroup>&>
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
};


}   // namespace SynapticEventKernel