#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
#include "codeStream.h"
#include "kernel.h"

//------------------------------------------------------------------------
// GPUStaticGrid
//------------------------------------------------------------------------
template<typename GroupType, typename... Arguments>
class GPUStaticGrid
{
public:
    //------------------------------------------------------------------------
    // IWriter
    //------------------------------------------------------------------------
    class IWriter
    {
    public:
        virtual void generateGlobals(CodeStream &os, const std::string &ftype, Arguments... arguments) const = 0;
        virtual void generateGroup(CodeStream &os, const GroupType &group, const std::string &ftype, Arguments... arguments) const = 0;
    };

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static void writeKernelCall(CodeStream &os, const KernelGPU<GroupType> *kernel, bool timingEnabled)
    {
        const unsigned int gridSizeBlocks = kernel->getMaxGridSizeBlocks();
        os << "// " << kernel->getKernelName() << " grid size = " << gridSizeBlocks << std::endl;
        os << CodeStream::OB(1131) << std::endl;

        // Declare threads and grid
        os << "dim3 threads(" << kernel->getBlockSize() << ", 1);" << std::endl;
        os << "dim3 grid(" << gridSizeBlocks << ", 1);" << std::endl;

        // Write code to record kernel start time
        // **TODO** correct names
        if(timingEnabled) {
            os << "cudaEventRecord(synapseStart);" << std::endl;
        }

        // Write call to kernel, passing in any extra global parameters and time
        os << kernel->getKernelName() << " <<<grid, threads>>>(";
        for(const auto &p : kernel->getExtraGlobalParameters()) {
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

    static void generateKernel(CodeStream &os, const KernelGPU<GroupType> *kernel, const IWriter *writer,
                               const std::string &ftype, Arguments... arguments)
    {
        os << "extern \"C\" __global__ void " << kernel->getKernelName() << "(";
        for (const auto &p : kernel->getExtraGlobalParameters()) {
            os << p.second << " " << p.first << ", ";
        }
        os << ftype << " t)" << std::endl; // end of synapse kernel header
        os << CodeStream::OB(75);

        os << "const unsigned int id = " << kernel->getBlockSize() << " * blockIdx.x + threadIdx.x;" << std::endl;

        // Generate globals
        writer->generateGlobals(os, ftype, arguments...);

        // Loop through the groups
        for(const auto &g : kernel->getGrid()) {
            os << "// group " << std::get<0>(g)->first << std::endl;
            os << "if ((id >= " << std::get<1>(g) << ") && (id < " << std::get<2>(g) << "))" << CodeStream::OB(77);
            os << "const unsigned int lid = id - " << std::get<1>(g) << ";" << std::endl;

            // Generate group code
            writer->generateGroup(os, std::get<0>(g)->second, ftype, arguments...);

            os << CodeStream::CB(77);
            os << std::endl;
        }

        os << CodeStream::CB(75);
    }
};