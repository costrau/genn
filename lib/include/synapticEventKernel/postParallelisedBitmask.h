#pragma once

// GeNN includes
#include "base.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::PostParallelisedBitmask
//----------------------------------------------------------------------------
namespace SynapticEventKernel
{
class PostParallelisedBitmask : public Base
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< How compatible is this kernel generator with this synapse group?
    //!< Ascending values indicate compatibility and negative numbers indicate incompatible
    virtual int getCompatibility(const SynapseGroup &sg) const override;

    //!<  Generate a kernel for simulating the specified subset
    //!<  of synapse groups and write it to the CodeStream
    virtual void generateKernel(CodeStream &os, bool isResetKernel,
                                unsigned int totalSynapseBlocks,
                                const std::map<std::string, NeuronGroup> &ngs,
                                const std::string &ftype) const override;

    //!< Get the name of the kernel (used to call it from runner)
    virtual std::string getKernelName() const override{ return "calcPostParallelisedBitmaskSynapses"; }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< Determine how many threads this synapse group
    //!< requires, not taking into account block size etc
    virtual unsigned int getNumThreads(const SynapseGroup &sg) const override;

private:
    //------------------------------------------------------------------------
    // Private API
    //------------------------------------------------------------------------
    void generateInnerLoop(CodeStream &os, const SynapseGroup &sg,
                           const string &postfix, const string &ftype) const;
};
}   // namespace SynapticEventKernel