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
    virtual void writeKernelCall(CodeStream &os, bool timingEnabled) const override
    {
        const unsigned int gridSizeBlocks = this->getMaxGridSizeBlocks();
        os << "// " << this->getKernelName() << " grid size = " << gridSizeBlocks << std::endl;
        os << CodeStream::OB(1131) << std::endl;

        // Declare threads and grid
        os << "dim3 threads(" << this->getBlockSize() << ", 1);" << std::endl;
        os << "dim3 grid(" << gridSizeBlocks << ", 1);" << std::endl;

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
        os << "t);" << std::endl;

        // Write code to record kernel stop time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStop);" << std::endl;
        }
        os << CodeStream::CB(1131) << std::endl;
    }

    virtual void generateKernel(CodeStream &os, Arguments... arguments, const std::string &ftype) const override
    {
        os << "extern \"C\" __global__ void " << this->getKernelName() << "(";
        for (const auto &p : this->getExtraGlobalParameters()) {
            os << p.second << " " << p.first << ", ";
        }
        os << ftype << " t)" << std::endl; // end of synapse kernel header
        os << CodeStream::OB(75);

        os << "const unsigned int id = " << this->getBlockSize() << " * blockIdx.x + threadIdx.x;" << std::endl;

        // Generate globals
        generateGlobals(os, ftype, arguments...);

        // Loop through the groups
        for(const auto &g : this->getGrid()) {
            os << "// group " << std::get<0>(g)->first << std::endl;
            os << "if ((id >= " << std::get<1>(g) << ") && (id < " << std::get<2>(g) << "))" << CodeStream::OB(77);
            os << "const unsigned int lid = id - " << std::get<1>(g) << ";" << std::endl;

            // Generate group code
            generateGroup(os, std::get<0>(g)->second, ftype, arguments...);

            os << CodeStream::CB(77);
            os << std::endl;
        }

        os << CodeStream::CB(75);
    }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void generateGlobals(CodeStream &os, const std::string &ftype, Arguments... arguments) const = 0;
    virtual void generateGroup(CodeStream &os, const typename BaseType::GroupType &sg, const std::string &ftype, Arguments... arguments) const = 0;
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
    virtual void writeKernelCall(CodeStream &os, const bool timingEnabled) const override
    {
        os << "// " << this->getKernelName() << std::endl;
        os << CodeStream::OB(1131) << std::endl;

        // Loop through each source neuron group and calculate padded size required to process it's spikes
        for(const auto &s : getSrcNeuronGroups()) {
            os << "const unsigned int padGlbSpkCnt" << s << " = (unsigned int)(ceil((double)spikeCount_" << s << " / (double)" << this->getBlockSize() << ") * (double)" << this->getBlockSize() << ");" << std::endl;
        }

        os << std::endl;
        for(unsigned int i = 0; i < this->getGrid().size(); i++) {
            const auto &g = this->getGrid()[i];
            os << "const unsigned int endId" << std::get<0>(g)->first << " = padGlbSpkCnt" << std::get<0>(g)->second.getSrcNeuronGroup()->getName();
            if(i > 0) {
                os << " + endId" << std::get<0>(this->getGrid()[i - 1])->first;
            }
            os << ";" << std::endl;
        }

        os << std::endl;

        // Declare threads
        os << "dim3 threads(" << this->getBlockSize() << ", 1);" << std::endl;

        // Declare grid based on sum of padded sizes
        os << "dim3 grid(endId" << std::get<0>(this->getGrid().back())->first << ", 1);" << std::endl;

        os << std::endl;

        // Write code to record kernel start time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStart);" << std::endl;
        }

        // Write call to kernel, passing in any extra global parameters and time
        os << this->getKernelName() << " <<<grid, threads>>>(";
        for(const auto &g : this->getGrid()) {
            os << "endId" << std::get<0>(g)->first << ", ";
        }

        for(const auto &p : this->getExtraGlobalParameters()) {
            os << p.first << ", ";
        }

        os << "t);" << std::endl;

        // Write code to record kernel stop time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStop);" << std::endl;
        }
        os << CodeStream::CB(1131) << std::endl;
    }

    virtual void generateKernel(CodeStream &os, Arguments... arguments, const std::string &ftype) const override
    {
        os << "extern \"C\" __global__ void " << this->getKernelName() << "(";
        for(const auto &g : this->getGrid()) {
            os << "unsigned int endId" << std::get<0>(g)->first << ", ";
        }

        for (const auto &p : this->getExtraGlobalParameters()) {
            os << p.second << " " << p.first << ", ";
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
                os << "if (id < endId" << std::get<0>(g)->first << ")" << CodeStream::OB(77);
                os << "const unsigned int lid = id;" << std::endl;
            }
            else {
                const std::string &prevSG = std::get<0>(this->getGrid()[i - 1])->first;
                os << "if ((id >= endId" << prevSG << ") && (id < endId" << std::get<0>(g)->first << "))" << CodeStream::OB(77);
                os << "const unsigned int lid = id - endId" << prevSG << ";" << std::endl;
            }

            // Generate group code
            generateGroup(os, std::get<0>(g)->second, ftype, arguments...);

            os << CodeStream::CB(77);
            os << std::endl;
        }

        os << CodeStream::CB(75);
    }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void generateGlobals(CodeStream &os, const std::string &ftype, Arguments... arguments) const = 0;
    virtual void generateGroup(CodeStream &os, const typename BaseType::GroupType &sg, const std::string &ftype, Arguments... arguments) const = 0;

private:
    std::set<std::string> getSrcNeuronGroups() const
    {
        std::set<std::string> srcNeuronGroups;
        for(const auto &g : this->getGrid()) {
            srcNeuronGroups.insert(std::get<0>(g)->second.getSrcNeuronGroup()->getName());
        }
        return srcNeuronGroups;
    }
};
