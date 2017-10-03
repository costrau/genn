#include "synapticEventKernel/preParallelisedSparse.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "codeStream.h"
#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"
#include "synapseMatrixType.h"
#include "utils.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::PreParallelisedSparse
//----------------------------------------------------------------------------
int SynapticEventKernel::PreParallelisedSparse::getCompatibility(const SynapseGroup &sg) const
{
    // Not compatible with non-presynaptic parallelism
    // **TODO** this can become automatic
    if(sg.getSpanType() != SynapseGroup::SpanType::PRESYNAPTIC) {
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
void SynapticEventKernel::PreParallelisedSparse::generateGlobals(CodeStream &os, const std::string &ftype,
                                                                 bool, unsigned int,
                                                                 const std::map<std::string, NeuronGroup>&) const
{
    // Global variables
    os << "unsigned int j, r;" << std::endl;
    os << "unsigned int ipost;" << std::endl;
    os << ftype << " addtoinSyn;" << std::endl;
    os << "unsigned int prePos; " << std::endl;
    os << "unsigned int npost; " << std::endl;


}
//----------------------------------------------------------------------------
void SynapticEventKernel::PreParallelisedSparse::generateGroup(CodeStream &os, const SynapseGroup &sg, const std::string &ftype,
                                                               bool isResetKernel, unsigned int totalSynapseBlocks,
                                                               const std::map<std::string, NeuronGroup> &ngs) const
{
    // Read delay slot if required
    StandardGeneratedSections::synapseReadDelaySlot(os, sg);

    // generate the code for processing spike-like events
    if (sg.isSpikeEventRequired()) {
        generateInnerLoop(os, sg, "Evnt", ftype);
    }

    // generate the code for processing true spike events
    if (sg.isTrueSpikeRequired()) {
        generateInnerLoop(os, sg, "", ftype);
    }
    os << std::endl;

    // If this is the reset kernel, insert reset kernel
    if (isResetKernel) {
        StandardGeneratedSections::synapseResetKernel(os, totalSynapseBlocks, ngs);
    }
}
//----------------------------------------------------------------------------
unsigned int SynapticEventKernel::PreParallelisedSparse::getMaxNumThreads(const SynapseGroup &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
void SynapticEventKernel::PreParallelisedSparse::generateInnerLoop(
    CodeStream &os, //!< output stream for code
    const SynapseGroup &sg,
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype) const
{
    const bool evnt = (postfix == "Evnt");
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());

    os << "if (lid < " ;
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "dd_glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[delaySlot])" << CodeStream::OB(102);
    }
    else {
        os << "dd_glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[0])" << CodeStream::OB(102);
    }

    if (!wu->getSimSupportCode().empty()) {
        os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
    }

    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "const int preInd = dd_glbSpk"  << postfix << sg.getSrcNeuronGroup()->getName();
        os << "[(delaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ") + lid];" << std::endl;
    }
    else {
        os << "const int preInd = dd_glbSpk"  << postfix << sg.getSrcNeuronGroup()->getName();
        os << "[lid];" << std::endl;
    }

    // Get index of start of row and its length from sparse matrix row indices
    os << "prePos = dd_indInG" << sg.getName() << "[preInd];" << std::endl;
    os << "npost = dd_indInG" << sg.getName() << "[preInd + 1] - prePos;" << std::endl;


    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << "if ";

        // code substitutions ----
        string eCode = wu->getEventThresholdConditionCode();
        StandardSubstitutions::weightUpdateThresholdCondition(eCode, sg,
                                                              wuDerivedParams, wuExtraGlobalParams,
                                                              "preInd", "i", "dd_", ftype);
        // end code substitutions ----
        os << "(" << eCode << ")";

        if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << ")";
        }
        os << CodeStream::OB(130);
    }


    os << "for (int i = 0; i < npost; ++i)" << CodeStream::OB(103);
    os << "ipost = dd_ind" <<  sg.getName() << "[prePos];" << std::endl;


// Code substitutions ----------------------------------------------------------------------------------
    string wCode = evnt ? wu->getEventCode() : wu->getSimCode();
    substitute(wCode, "$(t)", "t");

    // Substitute in correct atomic operation
    substitute(wCode, "$(updatelinsyn)", getFloatAtomicAdd(ftype) + "($(inSyn), $(addtoinSyn))");
    substitute(wCode, "$(inSyn)", "&dd_inSyn" + sg.getName() + "[ipost]");

    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[prePos]");
    }

    StandardSubstitutions::weightUpdateSim(wCode, sg,
                                           wuVars, wuDerivedParams, wuExtraGlobalParams,
                                           "preInd", "ipost", "dd_", ftype);
    // end code substitutions -------------------------------------------------------------------------

    os << wCode << std::endl;
    os << "prePos += 1;" << std::endl;

    os << CodeStream::CB(103);
    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << CodeStream::CB(130);
    }
    os << CodeStream::CB(102);
}
