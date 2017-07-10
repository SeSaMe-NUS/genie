#ifndef DEVICE_CODEC_FACTORY_H_
#define DEVICE_CODEC_FACTORY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "DeviceCodecs.h"
#include <genie/matching/match.h>
#include <genie/utility/Logger.h>

namespace GPUGenie {

enum COMPRESSION_TYPE {
    NO_COMPRESSION = 0, // make (bool)NO_COMPRESSION evaluate as false 
    COPY,
    DELTA,
    BP32,
    VARINT,
    COMP_BP32_COPY,
    COMP_BP32_VARINT,
    SERIAL_COPY_COPY,
    SERIAL_DELTA_COPY,
    SERIAL_DELTA_DELTA,
    SERIAL_DELTA_VARINT,
    SERIAL_DELTA_BP32,
    SERIAL_DELTA_COMP_BP32_COPY,
    SERIAL_DELTA_COMP_BP32_VARINT,
};

extern const COMPRESSION_TYPE DEFAULT_COMPRESSION_TYPE;
extern const COMPRESSION_TYPE LIGHTWEIGHT_COMPRESSION_TYPE;
extern const COMPRESSION_TYPE MIDDLEWEIGHT_COMPRESSION_TYPE;
extern const COMPRESSION_TYPE HEAVYWEIGHT_COMPRESSION_TYPE;

/**
 * Typedef IntegratedKernelPtr as a function pointer to instanced template of match_integrated
 */
class inv_compr_table;
typedef void (
        *MatchIntegratedFunPtr)(inv_compr_table&, std::vector<query>&, thrust::device_vector<genie::matching::data_t>&,
        thrust::device_vector<u32>&, int, int, thrust::device_vector<u32>&, thrust::device_vector<u32>&,
        thrust::device_vector<u32>&);


class DeviceCodecFactory {
public:

    static const std::map<std::string, COMPRESSION_TYPE> compressionTypesMap;

    static const std::map<COMPRESSION_TYPE, std::string> compressionNamesMap;
    
    static const std::map<COMPRESSION_TYPE, MatchIntegratedFunPtr> integratedKernelsMap;

    // TODO test that the class inheritance structure works well (e.g. by calling getName())
    static const std::map<COMPRESSION_TYPE, std::shared_ptr<DeviceIntegerCODEC>> codecInstancesMap;

    static const std::vector<COMPRESSION_TYPE> allCompressionTypes;

    static const std::vector<std::string> allCompressionNames;

    static std::string getCompressionName(COMPRESSION_TYPE type)
    {
        if (compressionNamesMap.find(type) == compressionNamesMap.end())
            return "Unknown-Compression-Type";
        return compressionNamesMap.at(type);
    }

    static COMPRESSION_TYPE getCompressionType(const std::string &name)
    {
        if (compressionTypesMap.find(name) == compressionTypesMap.end())
            return NO_COMPRESSION;
        return compressionTypesMap.at(name);
    }

    static std::shared_ptr<DeviceIntegerCODEC> getCodec(COMPRESSION_TYPE type)
    {
        if (codecInstancesMap.find(type) == codecInstancesMap.end())
        {
            Logger::log(Logger::ALERT, "Unknown codec requested (%d)!", (int)type);
            return nullCodec;
        }
        return codecInstancesMap.at(type);
    }

    static MatchIntegratedFunPtr getMatchingFunPtr(COMPRESSION_TYPE type)
    {
        if (integratedKernelsMap.find(type) == integratedKernelsMap.end())
        {
            Logger::log(Logger::ALERT, "Unknown integrated matching function pointer requested (%d)!", (int)type);
            return nullptr;
        }
        return integratedKernelsMap.at(type);
    }

protected:

    static const std::shared_ptr<DeviceIntegerCODEC> nullCodec;

};

}

#endif
