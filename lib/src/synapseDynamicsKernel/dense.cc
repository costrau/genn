#include "synapseDynamicsKernel/dense.h"

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
int SynapseDynamicsKernel::Dense::getCompatibility(const SynapseGroup &sg) const
{
    // Not compatible with non-dense matrices
    if(!(sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)) {
        return -1;
    }

    // Compatible
    return 1;
}
//----------------------------------------------------------------------------
void SynapseDynamicsKernel::Dense::generateGlobals(CodeStream &os, const std::string &ftype) const
{
    // Global variables
    os << "unsigned int id = " << getBlockSize() << " * blockIdx.x + threadIdx.x;" << std::endl;
    os << ftype << " addtoinSyn;" << std::endl;
    os << std::endl;
}
//----------------------------------------------------------------------------
void SynapseDynamicsKernel::Dense::generateGroup(CodeStream &os, const SynapseGroup &sg, const std::string &ftype) const
{
    const auto *wu = sg.getWUModel();

    // Read delay slot if required
    StandardGeneratedSections::synapseReadDelaySlot(os, sg);

    // Create iteration context to iterate over the variables and derived parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());

    string SDcode = wu->getSynapseDynamicsCode();
    substitute(SDcode, "$(t)", "t");

     os << "if (lid < " << sg.getSrcNeuronGroup()->getNumNeurons() * sg.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(25);
    os << "// all threads participate that can work on an existing synapse" << std::endl;
    if (!wu->getSynapseDynamicsSuppportCode().empty()) {
            os << " using namespace " << sg.getName() << "_weightupdate_synapseDynamics;" << std::endl;
    }
    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        // name substitute synapse var names in synapseDynamics code
        name_substitutions(SDcode, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[lid]");
    }

    const std::string postIdx = "lid %" + to_string(sg.getTrgNeuronGroup()->getNumNeurons());
    substitute(SDcode, "$(updatelinsyn)", getFloatAtomicAdd(ftype) + "(&$(inSyn), $(addtoinSyn))");
    substitute(SDcode, "$(inSyn)", "dd_inSyn" + sg.getName() + "[" + postIdx + "]");

    StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                "lid /" + to_string(sg.getTrgNeuronGroup()->getNumNeurons()),
                                                postIdx, "dd_", ftype);
    os << SDcode << std::endl;
    os << CodeStream::CB(25);
}
//----------------------------------------------------------------------------
unsigned int SynapseDynamicsKernel::Dense::getMaxNumThreads(const SynapseGroup &sg) const
{
    return ceil((double)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getTrgNeuronGroup()->getNumNeurons() / (double)getBlockSize()) * (double)getBlockSize();
}
