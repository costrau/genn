#include "synapticEventKernel/postParallelisedDense.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "codeStream.h"
#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"
#include "synapseMatrixType.h"

//----------------------------------------------------------------------------
// SynapticEventKernel::PostParallelisedDense
//----------------------------------------------------------------------------
int SynapticEventKernel::PostParallelisedDense::getCompatibility(const SynapseGroup &sg) const
{
    // Not compatible with non-postsynaptic parallelism
    // **TODO** this can become automatic
    if(sg.getSpanType() != SynapseGroup::SpanType::POSTSYNAPTIC) {
        return -1;
    }

    // Not compatible with non-dense matrices
    if(!(sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)) {
        return -1;
    }

    // Compatible
    return 1;
}
//----------------------------------------------------------------------------
void SynapticEventKernel::PostParallelisedDense::generateKernel(CodeStream &os, bool isResetKernel,
                                                                unsigned int totalSynapseBlocks,
                                                                const std::map<std::string, NeuronGroup> &ngs,
                                                                const std::string &ftype) const
{
    // synapse kernel header
    writeKernelDeclaration(os, ftype);

    // synapse kernel code
    os << CodeStream::OB(75);

    // Global variables
    os << "unsigned int id = " << getBlockSize() << " * blockIdx.x + threadIdx.x;" << std::endl;
    os << "unsigned int lmax, j, r;" << std::endl;
    os << "unsigned int ipost;" << std::endl;
    os << ftype << " addtoinSyn;" << std::endl;
    os << ftype << " linSyn;" << std::endl;

    // Do any of the synapse groups process true spikes
    if(std::any_of(getGrid().cbegin(), getGrid().cend(),
       [](const GridEntry &g){ return std::get<0>(g)->second.isTrueSpikeRequired(); }))
    {
        os << "__shared__ unsigned int shSpk[" << getBlockSize() << "];" << std::endl;
        os << "unsigned int lscnt, numSpikeSubsets;" << std::endl;
    }

    // Do any of the synapse groups process spike-like events
    if(std::any_of(getGrid().cbegin(), getGrid().cend(),
       [](const GridEntry &g){ return std::get<0>(g)->second.isSpikeEventRequired(); }))
    {
        os << "__shared__ unsigned int shSpkEvnt[" << getBlockSize() << "];" << std::endl;
        os << "unsigned int lscntEvnt, numSpikeSubsetsEvnt;" << std::endl;
    }

    // Loop through the synapse groups
    for(const auto &g : getGrid()) {
        const SynapseGroup &sg = std::get<0>(g)->second;

        os << "// synapse group " << std::get<0>(g)->first << std::endl;

        os << "if ((id >= " << std::get<1>(g) << ") && (id < " << std::get<2>(g) << "))" << CodeStream::OB(77);
        os << "const unsigned int lid = id - " << std::get<1>(g) << ";" << std::endl;

        if (sg.getSrcNeuronGroup()->isDelayRequired()) {
            os << "unsigned int delaySlot = (dd_spkQuePtr" << sg.getSrcNeuronGroup()->getName();
            os << " + " << (sg.getSrcNeuronGroup()->getNumDelaySlots() - sg.getDelaySteps());
            os << ") % " << sg.getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
        }

        os << "// only do this for existing neurons" << std::endl;
        os << "if (lid < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(80);
        os << "linSyn = dd_inSyn" << std::get<0>(g)->first << "[lid];" << std::endl;
        os << CodeStream::CB(80);

        // If spike-like events are processed, extract spike count
        if (sg.isSpikeEventRequired()) {
            os << "lscntEvnt = dd_glbSpkCntEvnt" << sg.getSrcNeuronGroup()->getName();
            if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                os << "[delaySlot];" << std::endl;
            }
            else {
                os << "[0];" << std::endl;
            }
            os << "numSpikeSubsetsEvnt = (lscntEvnt+" << getBlockSize() << "-1) / " << getBlockSize() << ";" << std::endl;
        }

        // If true spikes are processed, extract spike count
        if (sg.isTrueSpikeRequired()) {
            os << "lscnt = dd_glbSpkCnt" << sg.getSrcNeuronGroup()->getName();
            if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                os << "[delaySlot];" << std::endl;
            }
            else {
                os << "[0];" << std::endl;
            }
            os << "numSpikeSubsets = (lscnt+" << getBlockSize() << "-1) / " << getBlockSize() << ";" << std::endl;
        }

        // generate the code for processing spike-like events
        if (sg.isSpikeEventRequired()) {
            generateInnerLoop(os, sg, "Evnt", ftype);
        }

        // generate the code for processing true spike events
        if (sg.isTrueSpikeRequired()) {
            generateInnerLoop(os, sg, "", ftype);
        }
        os << std::endl;

        // Copy updated value back to main memory from register
        os << "// only do this for existing neurons" << std::endl;
        os << "if (lid < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(190);
        os << "dd_inSyn" << std::get<0>(g)->first << "[lid] = linSyn;" << std::endl;
        os << CodeStream::CB(190);

        // If this is the reset kernel, insert reset kernel
        if (isResetKernel) {
           StandardGeneratedSections::resetKernel(os, totalSynapseBlocks, ngs);
        }

        os << CodeStream::CB(77);
        os << std::endl;
    }
}
//----------------------------------------------------------------------------
unsigned int SynapticEventKernel::PostParallelisedDense::getPaddedSize(const SynapseGroup &sg) const
{
    return (unsigned int)(ceil((double)sg.getTrgNeuronGroup()->getNumNeurons() / (double)getBlockSize()) * (double)getBlockSize());
}
//----------------------------------------------------------------------------
void SynapticEventKernel::PostParallelisedDense::generateInnerLoop(
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

    os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << std::endl;
    os << "for (r = 0; r < numSpikeSubsets" << postfix << "; r++)" << CodeStream::OB(90);
    os << "if (r == numSpikeSubsets" << postfix << " - 1) lmax = ((lscnt" << postfix << "-1) % " << getBlockSize() << ") +1;" << std::endl;
    os << "else lmax = " << getBlockSize() << ";" << std::endl;
    os << "__syncthreads();" << std::endl;

    os << "if (threadIdx.x < lmax)" << CodeStream::OB(100);
    os << "j = dd_glbSpk" << postfix << sg.getSrcNeuronGroup()->getName() << "[" << sg.getOffsetPre() << "(r * " << getBlockSize() << ") + threadIdx.x];" << std::endl;

    // Copy spike ID into shared memory
    os << "shSpk" << postfix << "[threadIdx.x] = j;" << std::endl;
    os << CodeStream::CB(100);

    os << "__syncthreads();" << std::endl;


    os << "// loop through all incoming spikes" << std::endl;
    os << "for (j = 0; j < lmax; j++)" << CodeStream::OB(110);
    os << "// only work on existing neurons" << std::endl;
    os << "if (lid < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(120);

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

    os << "ipost = lid;" << std::endl;

    // Code substitutions ----------------------------------------------------------------------------------
    string wCode = (evnt ? wu->getEventCode() : wu->getSimCode());
    substitute(wCode, "$(t)", "t");

    substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
    substitute(wCode, "$(inSyn)", "linSyn");
    if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd,
                           sg.getName() + "[shSpk" + postfix + "[j] * " + to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + "+ ipost]");
    }

    StandardSubstitutions::weightUpdateSim(wCode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                           "shSpk" + postfix + "[j]", "ipost", "dd_", ftype);
    // end Code substitutions -------------------------------------------------------------------------
    os << wCode << std::endl;

    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << CodeStream::CB(130); // end if (eCode)
    }

    os << CodeStream::CB(120) << std::endl;

    os << CodeStream::CB(110) << std::endl;
    os << CodeStream::CB(90) << std::endl;
}
