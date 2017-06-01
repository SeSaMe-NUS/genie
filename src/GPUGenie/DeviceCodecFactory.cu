#include "DeviceBitPackingCodec.h"
#include "DeviceVarintCodec.h"
#include "DeviceCompositeCodec.h"
#include "DeviceSerialCodec.h"
#include "match_integrated.h"

#include "DeviceCodecFactory.h"

using namespace GPUGenie;

const COMPRESSION_TYPE GPUGenie::DEFAULT_COMPRESSION_TYPE = NO_COMPRESSION;
const COMPRESSION_TYPE GPUGenie::LIGHTWEIGHT_COMPRESSION_TYPE = BP32;
const COMPRESSION_TYPE GPUGenie::MIDDLEWEIGHT_COMPRESSION_TYPE = SERIAL_DELTA_BP32;
const COMPRESSION_TYPE GPUGenie::HEAVYWEIGHT_COMPRESSION_TYPE = SERIAL_DELTA_COMP_BP32_VARINT;


std::map<COMPRESSION_TYPE, shared_ptr<DeviceIntegerCODEC>> initCodecInstancesMap()
{
    std::map<COMPRESSION_TYPE, shared_ptr<DeviceIntegerCODEC>> map;

    map[NO_COMPRESSION] = shared_ptr<DeviceIntegerCODEC>(new DeviceCopyCodec());
    map[COPY] = shared_ptr<DeviceIntegerCODEC>(new DeviceCopyCodec());
    map[DELTA] = shared_ptr<DeviceIntegerCODEC>(new DeviceDeltaCodec());
    map[BP32] = shared_ptr<DeviceIntegerCODEC>(new DeviceBitPackingCodec());
    map[VARINT] = shared_ptr<DeviceIntegerCODEC>(new DeviceVarintCodec());
    map[COMP_BP32_COPY] = shared_ptr<DeviceIntegerCODEC>(new DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>());
    map[COMP_BP32_VARINT] = shared_ptr<DeviceIntegerCODEC>(new DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>());
    map[SERIAL_COPY_COPY] = shared_ptr<DeviceIntegerCODEC>(new DeviceSerialCodec<DeviceCopyCodec,DeviceCopyCodec>());
    map[SERIAL_DELTA_COPY] = shared_ptr<DeviceIntegerCODEC>(new DeviceSerialCodec<DeviceDeltaCodec,DeviceCopyCodec>());
    map[SERIAL_DELTA_DELTA] = shared_ptr<DeviceIntegerCODEC>(new DeviceSerialCodec<DeviceDeltaCodec,DeviceDeltaCodec>());
    map[SERIAL_DELTA_VARINT] = shared_ptr<DeviceIntegerCODEC>(new DeviceSerialCodec<DeviceDeltaCodec,DeviceVarintCodec>());
    map[SERIAL_DELTA_BP32] = shared_ptr<DeviceIntegerCODEC>(new DeviceSerialCodec<DeviceDeltaCodec,DeviceBitPackingCodec>());
    map[SERIAL_DELTA_COMP_BP32_COPY] = shared_ptr<DeviceIntegerCODEC>(new DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>());
    map[SERIAL_DELTA_COMP_BP32_VARINT] = shared_ptr<DeviceIntegerCODEC>(new DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>());

    return map;
}

const std::map<COMPRESSION_TYPE, shared_ptr<DeviceIntegerCODEC>> DeviceCodecFactory::codecInstancesMap = initCodecInstancesMap();

// NOTE: Template instantiation of match_integrated is in match_integrated.cu
//
// TODO: Figure out a way how to separate match_integrated instances into multiple files, similar to how Codecs
// templates are instantiated in their respective .cu files

std::map<COMPRESSION_TYPE, MatchIntegratedFunPtr> initIntegratedKernelsMap()
{
    std::map<COMPRESSION_TYPE, MatchIntegratedFunPtr> map;

    map[NO_COMPRESSION] = nullptr;
    map[COPY] = match_integrated<DeviceCopyCodec>;
    map[DELTA] = match_integrated<DeviceDeltaCodec>;
    map[BP32] = match_integrated<DeviceBitPackingCodec>;
    map[VARINT] = match_integrated<DeviceVarintCodec>;
    map[COMP_BP32_COPY] = match_integrated<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>;
    map[COMP_BP32_VARINT] = match_integrated<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>;
    map[SERIAL_COPY_COPY] = match_integrated<DeviceSerialCodec<DeviceCopyCodec,DeviceCopyCodec>>;
    map[SERIAL_DELTA_COPY] = match_integrated<DeviceSerialCodec<DeviceDeltaCodec,DeviceCopyCodec>>;
    map[SERIAL_DELTA_DELTA] = match_integrated<DeviceSerialCodec<DeviceDeltaCodec,DeviceDeltaCodec>>;
    map[SERIAL_DELTA_VARINT] = match_integrated<DeviceSerialCodec<DeviceDeltaCodec,DeviceVarintCodec>>;
    map[SERIAL_DELTA_BP32] = match_integrated<DeviceSerialCodec<DeviceDeltaCodec,DeviceBitPackingCodec>>;
    map[SERIAL_DELTA_COMP_BP32_COPY] = match_integrated<DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>>;
    map[SERIAL_DELTA_COMP_BP32_VARINT] = match_integrated<DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>>;

    return map;
}

const std::map<COMPRESSION_TYPE, MatchIntegratedFunPtr> DeviceCodecFactory::integratedKernelsMap = initIntegratedKernelsMap();

std::map<COMPRESSION_TYPE, std::string> initCompressionNamesMap()
{
    std::map<COMPRESSION_TYPE, std::string> map;    
    for (auto it = DeviceCodecFactory::codecInstancesMap.begin(); it != DeviceCodecFactory::codecInstancesMap.end(); it++)
        map[it->first] = it->second->name();
    return map;
}

const std::map<COMPRESSION_TYPE, std::string> DeviceCodecFactory::compressionNamesMap = initCompressionNamesMap();

std::map<std::string, COMPRESSION_TYPE> initCompressionTypesMap()
{
    std::map<std::string, COMPRESSION_TYPE> map;    
    for (auto it = DeviceCodecFactory::compressionNamesMap.begin(); it != DeviceCodecFactory::compressionNamesMap.end(); it++)
        map[it->second] = it->first;
    return map;
}

const std::map<std::string, COMPRESSION_TYPE> DeviceCodecFactory::compressionTypesMap = initCompressionTypesMap();

std::vector<std::string> initAllCompressionNames()
{
    std::vector<std::string> names;
    for (auto i = DeviceCodecFactory::compressionNamesMap.begin(); i != DeviceCodecFactory::compressionNamesMap.end(); ++i)
        names.push_back(i->second);
    return names;
}

const std::vector<std::string> DeviceCodecFactory::allCompressionNames = initAllCompressionNames();

std::vector<COMPRESSION_TYPE> initAllCompressionTypes()
{
    std::vector<COMPRESSION_TYPE> types;
    for (auto i = DeviceCodecFactory::integratedKernelsMap.begin(); i != DeviceCodecFactory::integratedKernelsMap.end(); ++i)
        types.push_back(i->first);
    return types;
}

const std::vector<COMPRESSION_TYPE> DeviceCodecFactory::allCompressionTypes = initAllCompressionTypes();

const shared_ptr<DeviceIntegerCODEC> DeviceCodecFactory::nullCodec = shared_ptr<DeviceIntegerCODEC>(nullptr);

