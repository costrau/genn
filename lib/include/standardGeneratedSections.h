#pragma once

// Standard includes
#include <map>
#include <string>

// GeNN includes
#include "codeGenUtils.h"
#include "newNeuronModels.h"
#include "standardSubstitutions.h"

// Forward declarations
class CodeStream;
class NeuronGroup;

//----------------------------------------------------------------------------
// Functions to generate standard sections of code for use across backends
//----------------------------------------------------------------------------
namespace StandardGeneratedSections
{
void neuronResetKernel(
    CodeStream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix);

void neuronLocalVarInit(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID);

void neuronLocalVarWrite(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID);

void neuronSpikeEventTest(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &localID,
    const std::string &ftype);

void resetKernel(
    CodeStream &os,
    const std::map<std::string, NeuronGroup> &ngs);
}   // namespace StandardGeneratedSections