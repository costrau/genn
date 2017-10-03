#include "synapticEventKernel/postParallelisedSparse.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "codeStream.h"
#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"
#include "synapseMatrixType.h"
#include "utils.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::PostParallelisedSparse
//----------------------------------------------------------------------------
int SynapticEventKernel::PostParallelisedSparse::getCompatibility(const SynapseGroup &sg) const
{
    // Not compatible with non-postsynaptic parallelism
    // **TODO** this can become automatic
    if(sg.getSpanType() != SynapseGroup::SpanType::POSTSYNAPTIC) {
        return -1;
    }

    // Not compatible with non-sparse matrices
    if(!(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE)) {
        return -1;
    }

    // Compatible
    return 1;
}
//----------------------------------------------------------------------------
void SynapticEventKernel::PostParallelisedSparse::generateGlobals(CodeStream &os, const std::string &ftype,
                                                                  bool, unsigned int,
                                                                  const std::map<std::string, NeuronGroup>&) const
{
    // Global variables
    os << "unsigned int id = " << getBlockSize() << " * blockIdx.x + threadIdx.x;" << std::endl;
    os << "unsigned int lmax, j, r;" << std::endl;
    os << "unsigned int ipost;" << std::endl;
    os << ftype << " addtoinSyn;" << std::endl;
    os << "unsigned int prePos; " << std::endl;
    os << "unsigned int npost; " << std::endl;

    // If any of the synapse groups are small enough, declare shared memory array to sum per-neuron outputs
    if(std::any_of(getGrid().cbegin(), getGrid().cend(),
        [this](const GridEntry &g){ return std::get<0>(g)->second.getTrgNeuronGroup()->getNumNeurons() <= getBlockSize(); }))
    {
        os << ftype << " linSyn;" << std::endl;
        os << "volatile __shared__ " << ftype << " shLg[" << getBlockSize() << "];" << std::endl;
    }

    // Do any of the synapse groups process true spikes
    if(std::any_of(getGrid().cbegin(), getGrid().cend(),
       [](const GridEntry &g){ return std::get<0>(g)->second.isTrueSpikeRequired(); }))
    {
        // If any of the true-spike processing synapse groups require indices for presynaptic variables
        if(std::any_of(getGrid().cbegin(), getGrid().cend(),
           [](const GridEntry &g){ return std::get<0>(g)->second.isTrueSpikeRequired() && std::get<0>(g)->second.arePreVarsRequiredForTrueSpike(); }))
        {
            os << "__shared__ unsigned int shSpk[" << getBlockSize() << "];" << std::endl;
        }
        os << "__shared__ unsigned int shSpkPrePos[" << getBlockSize() << "];" << std::endl;
        os << "__shared__ unsigned int shSpkNPost[" << getBlockSize() << "];" << std::endl;
    }

    // Do any of the synapse groups process spike-like events
    if(std::any_of(getGrid().cbegin(), getGrid().cend(),
       [](const GridEntry &g){ return std::get<0>(g)->second.isSpikeEventRequired(); }))
    {
        // If any of the spike-like event processing synapse groups require indices for presynaptic variables
        if(std::any_of(getGrid().cbegin(), getGrid().cend(),
           [](const GridEntry &g){ return std::get<0>(g)->second.isSpikeEventRequired() && std::get<0>(g)->second.arePreVarsRequiredForSpikeLikeEvent(); }))
        {
            os << "__shared__ unsigned int shSpkEvnt[" << getBlockSize() << "];" << std::endl;
        }
        os << "__shared__ unsigned int shSpkEvntPrePos[" << getBlockSize() << "];" << std::endl;
        os << "__shared__ unsigned int shSpkEvntNPost[" << getBlockSize() << "];" << std::endl;
    }
}
//----------------------------------------------------------------------------
void SynapticEventKernel::PostParallelisedSparse::generateGroup(CodeStream &os, const SynapseGroup &sg, const std::string &ftype,
                                                                bool isResetKernel, unsigned int totalSynapseBlocks,
                                                                const std::map<std::string, NeuronGroup> &ngs) const
{
    const bool useSharedMemory = (sg.getTrgNeuronGroup()->getNumNeurons() <= getBlockSize());

    // Read delay slot if required
    StandardGeneratedSections::synapseReadDelaySlot(os, sg);

    // Read out the number of true spikes and spike-like events and
    // determine how many blocks are required to process them in
    StandardGeneratedSections::synapseReadEventBlockCount(os, getBlockSize(), sg);

    // If we're using shared memory, copy current input into register
    if(useSharedMemory) {
        os << "// only do this for existing neurons" << std::endl;
        os << "if (lid < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(80);
        os << "linSyn = dd_inSyn" << sg.getName() << "[lid];" << std::endl;
        os << CodeStream::CB(80);
    }

    // generate the code for processing spike-like events
    if (sg.isSpikeEventRequired()) {
        generateInnerLoop(os, sg, useSharedMemory, "Evnt", ftype);
    }

    // generate the code for processing true spike events
    if (sg.isTrueSpikeRequired()) {
        generateInnerLoop(os, sg, useSharedMemory, "", ftype);
    }
    os << std::endl;

    // If we're using shared memory, add input transferred to register, back to main memory
    if(useSharedMemory) {
        os << "// only do this for existing neurons" << std::endl;
        os << "if (lid < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(190);
        os << "dd_inSyn" << sg.getName() << "[lid] = linSyn;" << std::endl;
        os << CodeStream::CB(190);
    }

    // If this is the reset kernel, insert reset kernel
    if (isResetKernel) {
        StandardGeneratedSections::synapseResetKernel(os, totalSynapseBlocks, ngs);
    }
}
//----------------------------------------------------------------------------
unsigned int SynapticEventKernel::PostParallelisedSparse::getMaxNumThreads(const SynapseGroup &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
void SynapticEventKernel::PostParallelisedSparse::generateInnerLoop(
    CodeStream &os, //!< output stream for code
    const SynapseGroup &sg,
    bool useSharedMemory,
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype) const
{
    const bool evnt = (postfix == "Evnt");
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());

    os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << std::endl;
    os << "for (r = 0; r < numSpikeSubsets" << postfix << "; r++)" << CodeStream::OB(90);
    os << "if (r == numSpikeSubsets" << postfix << " - 1) lmax = ((lscnt" << postfix << "-1) % " << getBlockSize() << ") +1;" << std::endl;
    os << "else lmax = " << getBlockSize() << ";" << std::endl;
    os << "__syncthreads();" << std::endl;

    os << "if (threadIdx.x < lmax)" << CodeStream::OB(100);
    os << "j = dd_glbSpk" << postfix << sg.getSrcNeuronGroup()->getName() << "[" << sg.getOffsetPre() << "(r * " << getBlockSize() << ") + threadIdx.x];" << std::endl;

    // If we need indices for presynaptic variables, copy spike ID into shared memory
    if(evnt ? sg.arePreVarsRequiredForSpikeLikeEvent() : sg.arePreVarsRequiredForTrueSpike()) {
        os << "shSpk" << postfix << "[threadIdx.x] = j;" << std::endl;
    }

    // Copy row offsets from sparse projection into shared memory
    os << "prePos = dd_indInG" << sg.getName() << "[j];" << std::endl;
    os << "shSpk" << postfix << "PrePos[threadIdx.x] = prePos;" << std::endl;
    os << "shSpk" << postfix << "NPost[threadIdx.x] = dd_indInG" << sg.getName() << "[j + 1] - prePos;" << std::endl;
    os << CodeStream::CB(100);

    // If input should be applied via shared memory, zero this
    // **NOTE** there are guaranteed to be enough threads to do this as maximum shared memory size IS block size
    if (useSharedMemory) {
        os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ") shLg[threadIdx.x] = 0;" << std::endl;
    }
    os << "__syncthreads();" << std::endl;


    os << "// loop through all incoming spikes" << std::endl;
    os << "for (j = 0; j < lmax; j++)" << CodeStream::OB(110);
    os << "// only work on existing neurons" << std::endl;
    os << "if (lid < " << sg.getMaxConnections() << ")" << CodeStream::OB(120);

    if (!wu->getSimSupportCode().empty()) {
        os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
    }
    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << "if ";
        // code substitutions ----
        string eCode = wu->getEventThresholdConditionCode();
        StandardSubstitutions::weightUpdateThresholdCondition(eCode, sg, wuDerivedParams, wuExtraGlobalParams,
                                                              "shSpkEvnt[j]", "ipost", "dd_", ftype);
        // end code substitutions ----
        os << "(" << eCode << ")";


        os << CodeStream::OB(130);
    }

    // Read offsets into sparse structure from shared memory
    os << "prePos = shSpk" << postfix << "PrePos[j];" << std::endl;
    os << "npost = shSpk" << postfix << "NPost[j];" << std::endl;

    // If this thread is within the sparse row
    os << "if (lid < npost)" << CodeStream::OB(140);
    os << "prePos += lid;" << std::endl;
    os << "ipost = dd_ind" << sg.getName() << "[prePos];" << std::endl;


    // Code substitutions ----------------------------------------------------------------------------------
    string wCode = (evnt ? wu->getEventCode() : wu->getSimCode());
    substitute(wCode, "$(t)", "t");

    // If we're using shared memory for output substitute in shared memory address
    if (useSharedMemory) {
        substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
        substitute(wCode, "$(inSyn)", "shLg[ipost]");
    }
    // Otherwise, substitute in correct atomic operation and address
    else {
        substitute(wCode, "$(updatelinsyn)", getFloatAtomicAdd(ftype) + "($(inSyn), $(addtoinSyn))");
        substitute(wCode, "$(inSyn)", "&dd_inSyn" + sg.getName() + "[ipost]");
    }

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[prePos]");
    }

    StandardSubstitutions::weightUpdateSim(wCode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                           "shSpk" + postfix + "[j]", "ipost", "dd_", ftype);
    // end Code substitutions -------------------------------------------------------------------------
    os << wCode << std::endl;

    os << CodeStream::CB(140); // end if (id < npost)

    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << CodeStream::CB(130); // end if (eCode)
    }

    os << CodeStream::CB(120) << std::endl;

    // If input should be applied via shared memory, copy out into linSyn output register
    if (useSharedMemory) {
        os << "__syncthreads();" << std::endl;
        os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(136); // need to write back results
        os << "linSyn += shLg[lid];" << std::endl;
        os << "shLg[lid] = 0;" << std::endl;
        os << CodeStream::CB(136) << std::endl;

        os << "__syncthreads();" << std::endl;
    }
    os << CodeStream::CB(110) << std::endl;
    os << CodeStream::CB(90) << std::endl;
}
