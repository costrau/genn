#include "synapsePostLearnKernel/dense.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "codeStream.h"
#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"
#include "synapseMatrixType.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::Dense
//----------------------------------------------------------------------------
int SynapsePostLearnKernel::Dense::getCompatibility(const SynapseGroup &sg) const
{
    // Not compatible with non-dense matrices
    if(!(sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)) {
        return -1;
    }

    // Compatible
    return 1;
}
//----------------------------------------------------------------------------
void SynapsePostLearnKernel::Dense::generateGlobals(CodeStream &os, const std::string &,
                                                    bool, unsigned int,
                                                    const std::map<std::string, NeuronGroup>&) const
{
    // Global variables
    os << "unsigned int lmax, j, r;" << std::endl;
    os << "__shared__ unsigned int shSpk[" << getBlockSize() << "];" << std::endl;
}
//----------------------------------------------------------------------------
void SynapsePostLearnKernel::Dense::generateGroup(CodeStream &os, const SynapseGroup &sg, const std::string &ftype,
                                                  bool isResetKernel, unsigned int totalPostLearnBlocks,
                                                  const std::map<std::string, NeuronGroup> &ngs) const
{
    const auto *wu = sg.getWUModel();

    // Read delay slot if required
    StandardGeneratedSections::synapseReadDelaySlot(os, sg);

    if (sg.getTrgNeuronGroup()->isDelayRequired() && sg.getTrgNeuronGroup()->isTrueSpikeRequired()) {
            os << "const unsigned int lscnt = dd_glbSpkCnt" << sg.getTrgNeuronGroup()->getName() << "[dd_spkQuePtr" << sg.getTrgNeuronGroup()->getName() << "];" << std::endl;
    }
    else {
        os << "const unsigned int lscnt = dd_glbSpkCnt" << sg.getTrgNeuronGroup()->getName() << "[0];" << std::endl;
    }

    os << "const unsigned int numSpikeSubsets = (lscnt+" << getBlockSize()-1 << ") / " << getBlockSize() << ";" << std::endl;
    os << "for (r = 0; r < numSpikeSubsets; r++)" << CodeStream::OB(230);
    os << "if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % " << getBlockSize() << ")+1;" << std::endl;
    os << "else lmax = " << getBlockSize() << ";" << std::endl;

    const string offsetTrueSpkPost = sg.getTrgNeuronGroup()->isTrueSpikeRequired() ? sg.getOffsetPost("dd_") : "";

    os << "if (threadIdx.x < lmax)" << CodeStream::OB(240);
    os << "shSpk[threadIdx.x] = dd_glbSpk" << sg.getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "(r * " << getBlockSize() << ") + threadIdx.x];" << std::endl;
    os << CodeStream::CB(240);

    os << "__syncthreads();" << std::endl;
    os << "// only work on existing neurons" << std::endl;
    os << "if (lid < " << sg.getSrcNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(250);
    os << "// loop through all incoming spikes for learning" << std::endl;
    os << "for (j = 0; j < lmax; j++)" << CodeStream::OB(260) << std::endl;

     if (!wu->getLearnPostSupportCode().empty()) {
        os << " using namespace " << sg.getName() << "_weightupdate_simLearnPost;" << std::endl;
    }

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());

    string code = wu->getLearnPostCode();
    substitute(code, "$(t)", "t");
    // Code substitutions ----------------------------------------------------------------------------------
    name_substitutions(code, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[lid * " + to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + " + shSpk[j]]");
    StandardSubstitutions::weightUpdatePostLearn(code, sg, wuDerivedParams, wuExtraGlobalParams,
                                                 "lid", "shSpk[j]", "dd_", ftype);
    // end Code substitutions -------------------------------------------------------------------------
    os << code << std::endl;
    os << CodeStream::CB(260);
    os << CodeStream::CB(250);
    os << CodeStream::CB(230);
    if (isResetKernel) {
        StandardGeneratedSections::synapseResetKernel(os, totalPostLearnBlocks, ngs);
    }
}
//----------------------------------------------------------------------------
unsigned int SynapsePostLearnKernel::Dense::getMaxNumThreads(const SynapseGroup &sg) const
{
    return ceil((double)sg.getSrcNeuronGroup()->getNumNeurons() / (double)getBlockSize()) * (double)getBlockSize();
}
