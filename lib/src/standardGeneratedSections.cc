#include "standardGeneratedSections.h"

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

//----------------------------------------------------------------------------
// StandardGeneratedSections
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronResetKernel(
    CodeStream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix)
{
    if (ng.isDelayRequired()) { // with delay
        os << devPrefix << "spkQuePtr" << ng.getName() << " = (" << devPrefix << "spkQuePtr" << ng.getName() << " + 1) % " << ng.getNumDelaySlots() << ";" << std::endl;
        if (ng.isSpikeEventRequired()) {
            os << devPrefix << "glbSpkCntEvnt" << ng.getName() << "[" << devPrefix << "spkQuePtr" << ng.getName() << "] = 0;" << std::endl;
        }
        if (ng.isTrueSpikeRequired()) {
            os << devPrefix << "glbSpkCnt" << ng.getName() << "[" << devPrefix << "spkQuePtr" << ng.getName() << "] = 0;" << std::endl;
        }
        else {
            os << devPrefix << "glbSpkCnt" << ng.getName() << "[0] = 0;" << std::endl;
        }
    }
    else { // no delay
        if (ng.isSpikeEventRequired()) {
            os << devPrefix << "glbSpkCntEvnt" << ng.getName() << "[0] = 0;" << std::endl;
        }

        if (ng.isTrueSpikeRequired()) {
            os << devPrefix << "glbSpkCnt" << ng.getName() << "[0] = 0;" << std::endl;
        }
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronLocalVarInit(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID)
{
    for(const auto &v : nmVars.container) {
        os << v.second << " l" << v.first << " = ";
        os << devPrefix << v.first << ng.getName() << "[";
        if (ng.isVarQueueRequired(v.first) && ng.isDelayRequired()) {
            os << "(delaySlot * " << ng.getNumNeurons() << ") + ";
        }
        os << localID << "];" << std::endl;
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronLocalVarWrite(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID)
{
    // store the defined parts of the neuron state into the global state variables dd_V etc
   for(const auto &v : nmVars.container) {
        if (ng.isVarQueueRequired(v.first)) {
            os << devPrefix << v.first << ng.getName() << "[" << ng.getQueueOffset(devPrefix) << localID << "] = l" << v.first << ";" << std::endl;
        }
        else {
            os << devPrefix << v.first << ng.getName() << "[" << localID << "] = l" << v.first << ";" << std::endl;
        }
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronSpikeEventTest(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &,
    const std::string &ftype)
{
    // Create local variable
    os << "bool spikeLikeEvent = false;" << std::endl;

    // Loop through outgoing synapse populations that will contribute to event condition code
    for(const auto &spkEventCond : ng.getSpikeEventCondition()) {
        // Replace of parameters, derived parameters and extraglobalsynapse parameters
        string eCode = spkEventCond.first;

        // code substitutions ----
        substitute(eCode, "$(id)", "n");
        StandardSubstitutions::neuronSpikeEventCondition(eCode, ng, nmVars, nmExtraGlobalParams, ftype);

        // Open scope for spike-like event test
        os << CodeStream::OB(31);

        // Use synapse population support code namespace if required
        if (!spkEventCond.second.empty()) {
            os << " using namespace " << spkEventCond.second << ";" << std::endl;
        }

        // Combine this event threshold test with
        os << "spikeLikeEvent |= (" << eCode << ");" << std::endl;

        // Close scope for spike-like event test
        os << CodeStream::CB(31);
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::resetKernel(
    CodeStream &os,
    const std::map<std::string, NeuronGroup> &ngs)
{
    os << "__syncthreads();" << std::endl;
    os << "if (threadIdx.x == 0)" << CodeStream::OB(200);
    os << "j = atomicAdd((unsigned int *) &d_done, 1);" << std::endl;
    os << "if (j == " << numSynapseBlocks - 1 << ")" << CodeStream::OB(210);

    for(const auto &n : model.getNeuronGroups()) {
        neuronResetKernel(os, n.second, "dd_");
    }
    os << "d_done = 0;" << std::endl;

    os << CodeStream::CB(210); // end "if (j == " << numOfBlocks - 1 << ")"
    os << CodeStream::CB(200); // end "if (threadIdx.x == 0)"
}