#pragma once

// Forward declarations
class ModelSpecInternal;
class SynapseGroupInternal;

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::Base
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
namespace PresynapticUpdateStrategy
{
class Base
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Get the stride i.e. the row length with which synaptic data structures should be allocated
    virtual size_t getStride(const SynapseGroupInternal &sg) const = 0;

    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg) const = 0;

    //! Can this presynaptic update strategy be used to simulate this synapse group?
    virtual bool canSimulate(const SynapseGroupInternal &sg) const = 0;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const SynapseGroupInternal &sg) const = 0;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const SynapseGroupInternal &sg) const = 0;

    //! Generate presynaptic update code
    virtual void generateCode(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, const Substitutions &popSubs, bool trueSpike,
                              SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const = 0;
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PreSpan
//--------------------------------------------------------------------------
//! Presynaptic parallelism
class PreSpan : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the stride i.e. the row length with which synaptic data structures should be allocated
    virtual size_t getStride(const SynapseGroupInternal &sg) const override;

    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg) const override;

    //! Can this presynaptic update strategy be used to simulate this synapse group?
    virtual bool canSimulate(const SynapseGroupInternal &sg) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const SynapseGroupInternal &sg) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const SynapseGroupInternal &sg) const override;

    //! Generate presynaptic update code
    virtual void generateCode(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, const Substitutions &popSubs, bool trueSpike,
                              SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const override;
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpan
//--------------------------------------------------------------------------
//! Postsynaptic parallelism
class PostSpan : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the stride i.e. the row length with which synaptic data structures should be allocated
    virtual size_t getStride(const SynapseGroupInternal &sg) const override;

    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg) const override;

    //! Can this presynaptic update strategy be used to simulate this synapse group?
    virtual bool canSimulate(const SynapseGroupInternal &sg) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const SynapseGroupInternal &sg) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const SynapseGroupInternal &sg) const override;

    //! Generate presynaptic update code
    virtual void generateCode(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, const Substitutions &popSubs, bool trueSpike,
                              SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const override;
};
}   // namespace PresynapticUpdateStrategy
}   // namespace CUDA
}   // namespace CodeGenerator