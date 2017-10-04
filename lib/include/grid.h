#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
#include "codeStream.h"
#include "kernel.h"
#include "synapseGroup.h"

//------------------------------------------------------------------------
// GPUStaticGrid
//------------------------------------------------------------------------
template<typename BaseType, typename... Arguments>
class GPUStaticGrid : public BaseType
{
public:
    //------------------------------------------------------------------------
    // KernelGPU virtuals
    //------------------------------------------------------------------------
    virtual void writeKernelCall(CodeStream &os, bool isResetKernel, bool timingEnabled) const override
    {
        os << "// " << this->getKernelName() << std::endl;
        os << CodeStream::OB(1131) << std::endl;

        // Declare threads and grid
        os << "dim3 threads(" << this->getBlockSize() << ", 1);" << std::endl;
        os << "dim3 grid(gridSize" << this->getKernelName() << ", 1);" << std::endl;

        // Write code to record kernel start time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStart);" << std::endl;
        }

        // Write call to kernel, passing in any extra global parameters and time
        os << this->getKernelName() << " <<<grid, threads>>>(";
        for(const auto &p : this->getExtraGlobalParameters()) {
            os << p.first << ", ";
        }

        // Add reset block count parameter if required
        if(isResetKernel) {
            os << "resetBlockCount, ";
        }

        os << "t);" << std::endl;

        // Write code to record kernel stop time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStop);" << std::endl;
        }
        os << CodeStream::CB(1131) << std::endl;
    }

    virtual void generateKernel(CodeStream &os, const std::string &ftype, bool isResetKernel, Arguments... arguments) const override
    {
        os << "extern \"C\" __global__ void " << this->getKernelName() << "(";
        for (const auto &p : this->getExtraGlobalParameters()) {
            os << p.second << " " << p.first << ", ";
        }

        // Add reset block count parameter if required
        if(isResetKernel) {
            os << "unsigned int resetBlockCount, ";
        }

        os << ftype << " t)" << std::endl; // end of synapse kernel header
        os << CodeStream::OB(75);

        os << "const unsigned int id = " << this->getBlockSize() << " * blockIdx.x + threadIdx.x;" << std::endl;

        // Generate globals
        generateGlobals(os, ftype, arguments...);

        // Loop through the groups
        for(unsigned int i = 0; i < this->getGrid().size(); i++) {
            const auto &g = this->getGrid()[i];

            os << "// group " << std::get<0>(g)->first << std::endl;
            if(i == 0) {
                os << "if (id < " << std::get<2>(g) << ")" << CodeStream::OB(77);
                os << "const unsigned int lid = id;" << std::endl;
            }
            else {
                os << "if ((id >= " << std::get<1>(g) << ") && (id < " << std::get<2>(g) << "))" << CodeStream::OB(77);
                os << "const unsigned int lid = id - " << std::get<1>(g) << ";" << std::endl;
            }

            // Generate group code
            generateGroup(os, std::get<0>(g)->second, ftype, isResetKernel, arguments...);

            os << CodeStream::CB(77);
            os << std::endl;
        }

        os << CodeStream::CB(75);
    }

    virtual void writePreamble(CodeStream &os) const override
    {
        os << "const unsigned int gridSize" << this->getKernelName() << " = " << this->getMaxGridSizeBlocks() << ";" << std::endl;
    }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void generateGlobals(CodeStream &os, const std::string &ftype, Arguments... arguments) const = 0;
    virtual void generateGroup(CodeStream &os, const typename BaseType::GroupType &sg,
                               const std::string &ftype, bool isResetKernel, Arguments... arguments) const = 0;
};

//------------------------------------------------------------------------
// GPUDynamicSpikeGrid
//------------------------------------------------------------------------
template<typename BaseType, typename... Arguments>
class GPUDynamicSpikeGrid : public BaseType
{
public:
    //------------------------------------------------------------------------
    // KernelGPU virtuals
    //------------------------------------------------------------------------
    virtual void writeKernelCall(CodeStream &os, bool isResetKernel, bool timingEnabled) const override
    {
        os << "// " << this->getKernelName() << " - kernel call" << std::endl;
        os << CodeStream::OB(1131) << std::endl;

        // Declare threads
        os << "dim3 threads(" << this->getBlockSize() << ", 1);" << std::endl;

        // Declare grid based on the end thread id of the last group in grid
        os << "dim3 grid(gridSize" << this->getKernelName() << ", 1);" << std::endl;

        os << std::endl;

        // Write code to record kernel start time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStart);" << std::endl;
        }

        // Write call to kernel, passing in end thread id of
        // each synapse group and extra global parameters
        os << this->getKernelName() << " <<<grid, threads>>>(";
        for(const auto &g : this->getGrid()) {
            os << "endId" << std::get<0>(g)->first << ", ";
        }
        for(const auto &p : this->getExtraGlobalParameters()) {
            os << p.first << ", ";
        }

        // Add reset block count parameter if required
        if(isResetKernel) {
            os << "resetBlockCount, ";
        }

        os << "t);" << std::endl;

        // Write code to record kernel stop time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStop);" << std::endl;
        }
        os << CodeStream::CB(1131) << std::endl;
    }

    virtual void generateKernel(CodeStream &os, const std::string &ftype, bool isResetKernel, Arguments... arguments) const override
    {
        // Write function declaration with parameters for the end
        // thread id of each synapse group and any extra gloval parameters
        os << "extern \"C\" __global__ void " << this->getKernelName() << "(";
        for(const auto &g : this->getGrid()) {
            os << "unsigned int endId" << std::get<0>(g)->first << ", ";
        }
        for (const auto &p : this->getExtraGlobalParameters()) {
            os << p.second << " " << p.first << ", ";
        }

        // Add reset block count parameter if required
        if(isResetKernel) {
            os << "unsigned int resetBlockCount, ";
        }

        os << ftype << " t)" << std::endl; // end of synapse kernel header
        os << CodeStream::OB(75);

        os << "const unsigned int id = " << this->getBlockSize() << " * blockIdx.x + threadIdx.x;" << std::endl;

        // Generate globals
        generateGlobals(os, ftype, arguments...);

        // Loop through the groups
        for(unsigned int i = 0; i < this->getGrid().size(); i++) {
            const auto &g = this->getGrid()[i];
            const std::string &name = std::get<0>(g)->first ;

            // If this is first group, check if thread id is less than end thread id of group
            os << "// group " << name << std::endl;
            if(i == 0) {
                os << "if (id < endId" << name << ")" << CodeStream::OB(77);
                os << "const unsigned int lid = id;" << std::endl;
            }
            // Otherwise, check if thread id is between end thread ids of preceding group and this one
            else {
                const std::string &prevName = std::get<0>(this->getGrid()[i - 1])->first;
                os << "if ((id >= endId" << prevName << ") && (id < endId" << name << "))" << CodeStream::OB(77);
                os << "const unsigned int lid = id - endId" << prevName << ";" << std::endl;
            }

            // Generate group code
            generateGroup(os, std::get<0>(g)->second, ftype, isResetKernel, arguments...);

            os << CodeStream::CB(77);
            os << std::endl;
        }

        os << CodeStream::CB(75);
    }

    virtual void writePreamble(CodeStream &os) const override
    {
        os << "// " << this->getKernelName() << " - kernel preamble" << std::endl;
        for(const auto &g : this->getGrid()) {
            os << "unsigned int endId" << std::get<0>(g)->first << ";" << std::endl;
        }

        os << CodeStream::OB(1131) << std::endl;

        // Loop through each source neuron group and calculate padded size required to process it's spikes
        std::set<const NeuronGroup*> requiredNeuronGroupSpikeCounts;
        getRequiredNeuronGroupSpikeCounts(requiredNeuronGroupSpikeCounts);
        for(const auto s : requiredNeuronGroupSpikeCounts) {
            os << "const unsigned int padGlbSpkCnt" << s->getName() << " = (unsigned int)(ceil((double)spikeCount_" << s->getName() << " / (double)" << this->getBlockSize() << ") * (double)" << this->getBlockSize() << ");" << std::endl;
        }
        os << std::endl;

        // Loop through each synapse group in grid
        for(unsigned int i = 0; i < this->getGrid().size(); i++) {
            const auto &g = this->getGrid()[i];
            const std::string &name = std::get<0>(g)->first;
            const std::string &srcNeuronName = std::get<0>(g)->second.getSrcNeuronGroup()->getName();

            // Calculate the end thread id of this synapse group by adding its padded size to
            // the padded size of the previous group (if this isn't the first synapse group)
            os << "endId" << name << " = padGlbSpkCnt" << srcNeuronName;
            if(i > 0) {
                os << " + endId" << std::get<0>(this->getGrid()[i - 1])->first;
            }
            os << ";" << std::endl;
        }
        os << CodeStream::CB(1131) << std::endl;

        os << std::endl;
        const std::string &lastName = std::get<0>(this->getGrid().back())->first;
        os << "const int gridSize" << this->getKernelName() << " = ceil((float)endId" << lastName << " / " << this->getBlockSize() << ");" << std::endl;
        os << std::endl;
    }

    //------------------------------------------------------------------------
    // SynapticEventKernel::BaseGPU virtuals
    //------------------------------------------------------------------------
    virtual void getRequiredNeuronGroupSpikeCounts(std::set<const NeuronGroup*> &neuronGroups) const override
    {
        for(const auto &g : this->getGrid()) {
            neuronGroups.insert(std::get<0>(g)->second.getSrcNeuronGroup());
        }
    }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void generateGlobals(CodeStream &os, const std::string &ftype, Arguments... arguments) const = 0;
    virtual void generateGroup(CodeStream &os, const typename BaseType::GroupType &sg,
                               const std::string &ftype, bool isResetKernel, Arguments... arguments) const = 0;

};
