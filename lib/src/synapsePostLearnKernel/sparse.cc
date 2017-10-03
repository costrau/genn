#include "synapsePostLearnKernel/sparse.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "codeStream.h"
#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"
#include "synapseMatrixType.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::Sparse
//----------------------------------------------------------------------------
int SynapsePostLearnKernel::Sparse::getCompatibility(const SynapseGroup &sg) const
{
    // Not compatible with non-dense matrices
    if(!(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE)) {
        return -1;
    }

    // Compatible
    return 1;
}
//----------------------------------------------------------------------------
void SynapsePostLearnKernel::Sparse::generateGlobals(CodeStream &os, const std::string &,
                                                    bool, unsigned int,
                                                    const std::map<std::string, NeuronGroup>&) const
{
    // Global variable
    os << "unsigned int id = " << getBlockSize() << " * blockIdx.x + threadIdx.x;" << std::endl;
    os << "unsigned int lmax, j, r;" << std::endl;

    if(std::any_of(getGrid().cbegin(), getGrid().cend(),
        [](const GridEntry &g){ return std::get<0>(g)->second.arePostVarsRequiredForPostLearning(); }))
    {
        os << "__shared__ unsigned int shSpk[" << getBlockSize() << "];" << std::endl;
    }
    os << "__shared__ unsigned int shSpkPrePos[" << getBlockSize() << "];" << std::endl;
    os << "__shared__ unsigned int shSpkNPre[" << getBlockSize() << "];" << std::endl;
}
//----------------------------------------------------------------------------
void SynapsePostLearnKernel::Sparse::generateGroup(CodeStream &os, const SynapseGroup &sg, const std::string &ftype,
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
    os << "j = dd_glbSpk" << sg.getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "(r * " << getBlockSize() << ") + threadIdx.x];" << std::endl;

    // If we need indices for presynaptic variables, copy spike ID into shared memory
    if(sg.arePostVarsRequiredForPostLearning()) {
        os << "shSpk[threadIdx.x] = j;" << std::endl;
    }

    // Copy row offsets from sparse projection into shared memory
    os << "const unsigned int iprePos = dd_revIndInG" << sg.getName() << "[j];" << std::endl;
    os << "shSpkPrePos[threadIdx.x] = iprePos;" << std::endl;
    os << "shSpkNPre[threadIdx.x] = dd_revIndInG" << sg.getName() << "[j + 1] - iprePos;" << std::endl;

    os << CodeStream::CB(240);

    os << "__syncthreads();" << std::endl;
    os << "// only work on existing neurons" << std::endl;
    os << "if (lid < " << sg.getSrcNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(250);
    os << "// loop through all incoming spikes for learning" << std::endl;
    os << "for (j = 0; j < lmax; j++)" << CodeStream::OB(260) << std::endl;

    // Read offsets into sparse structure from shared memory
    os << "unsigned int iprePos = shSpkPrePos[j];" << std::endl;
    os << "const unsigned int npre = shSpkNPre[j];" << std::endl;

    os << "if (lid < npre)" << CodeStream::OB(1540);
    os << "iprePos += lid;" << std::endl;

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
    name_substitutions(code, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[dd_remap" + sg.getName() + "[iprePos]]");
    StandardSubstitutions::weightUpdatePostLearn(code, sg, wuDerivedParams, wuExtraGlobalParams,
                                                 "dd_revInd" + sg.getName() + "[iprePos]", "shSpk[j]", "dd_", ftype);
    // end Code substitutions -------------------------------------------------------------------------
    os << code << std::endl;
    os << CodeStream::CB(1540);
    os << CodeStream::CB(260);
    os << CodeStream::CB(250);
    os << CodeStream::CB(230);
    if (isResetKernel) {
        StandardGeneratedSections::synapseResetKernel(os, totalPostLearnBlocks, ngs);
    }
}
//----------------------------------------------------------------------------
unsigned int SynapsePostLearnKernel::Sparse::getMaxNumThreads(const SynapseGroup &sg) const
{
    return ceil((double)sg.getSrcNeuronGroup()->getNumNeurons() / (double)getBlockSize()) * (double)getBlockSize();
}
