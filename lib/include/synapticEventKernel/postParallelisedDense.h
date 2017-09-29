#pragma once

// GeNN includes
#include "base.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::PostParallelisedDense
//----------------------------------------------------------------------------
namespace SynapticEventKernel
{
class PostParallelisedDense : public Base
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< How compatible is this kernel generator with this synapse group?
    //!< Ascending values indicate compatibility and negative numbers indicate incompatible
    int getCompatibility(const SynapseGroup &sg) const override;

    //!<  Generate a kernel for simulating the specified subset
    //!<  of synapse groups and write it to the CodeStream
    void generateKernel(CodeStream &os, bool isResetKernel) const override;

    //!< Get the name of the kernel (used to call it from runner)
    std::string getKernelName() const override{ return "calcPostParallelisedDenseSynapses"; }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //!< Determine how many threads this synapse group
    //!< requires, taking into account block size etc
    virtual unsigned int getPaddedSize(const SynapseGroup &sg) const override;

};
}   // namespace SynapticEventKernel