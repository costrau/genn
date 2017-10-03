#include "synapseDynamicsKernel/sparse.h"

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
int SynapseDynamicsKernel::Sparse::getCompatibility(const SynapseGroup &sg) const
{
    // Not compatible with non-sparse matrices
    if(!(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE)) {
        return -1;
    }

    // Compatible
    return 1;
}
//----------------------------------------------------------------------------
void SynapseDynamicsKernel::Sparse::generateGlobals(CodeStream &os, const std::string &ftype) const
{
    // Global variables
    os << ftype << " addtoinSyn;" << std::endl;
    os << std::endl;
}
//----------------------------------------------------------------------------
void SynapseDynamicsKernel::Sparse::generateGroup(CodeStream &os, const SynapseGroup &sg, const std::string &ftype) const
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

    os << "if (lid < dd_indInG" << sg.getName() << "[" << sg.getSrcNeuronGroup()->getNumNeurons() << "])" << CodeStream::OB(25);
    os << "// all threads participate that can work on an existing synapse" << std::endl;
    if (!wu->getSynapseDynamicsSuppportCode().empty()) {
        os << " using namespace " << sg.getName() << "_weightupdate_synapseDynamics;" << std::endl;
    }

    // name substitute synapse var names in synapseDynamics code
    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(SDcode, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[lid]");
    }

    const std::string postIdx = "dd_ind" + sg.getName() + "[lid]";
    substitute(SDcode, "$(updatelinsyn)", getFloatAtomicAdd(ftype) + "(&$(inSyn), $(addtoinSyn))");
    substitute(SDcode, "$(inSyn)", "dd_inSyn" + sg.getName() + "[" + postIdx + "]");

    StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                "dd_preInd" + sg.getName() +"[lid]",
                                                postIdx, "dd_", ftype);
    os << SDcode << std::endl;
    os << CodeStream::CB(25);
}
//----------------------------------------------------------------------------
unsigned int SynapseDynamicsKernel::Sparse::getMaxNumThreads(const SynapseGroup &sg) const
{
    return ceil((double)sg.getSrcNeuronGroup()->getNumNeurons() * sg.getMaxConnections() / (double)getBlockSize()) * (double)getBlockSize();
}
