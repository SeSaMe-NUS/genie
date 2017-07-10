#include "DeviceBitPackingCodec.h"
#include "DeviceVarintCodec.h"
#include "DeviceCompositeCodec.h"
#include "DeviceSerialCodec.h"
#include <genie/matching/match_integrated.h>

#include "DeviceCodecFactory.h"

using namespace GPUGenie;
using namespace std;

const COMPRESSION_TYPE GPUGenie::DEFAULT_COMPRESSION_TYPE = NO_COMPRESSION;
const COMPRESSION_TYPE GPUGenie::LIGHTWEIGHT_COMPRESSION_TYPE = BP32;
const COMPRESSION_TYPE GPUGenie::MIDDLEWEIGHT_COMPRESSION_TYPE = SERIAL_DELTA_BP32;
const COMPRESSION_TYPE GPUGenie::HEAVYWEIGHT_COMPRESSION_TYPE = SERIAL_DELTA_COMP_BP32_VARINT;


map<COMPRESSION_TYPE, shared_ptr<DeviceIntegerCODEC>> initCodecInstancesMap()
{
    map<COMPRESSION_TYPE, shared_ptr<DeviceIntegerCODEC>> map;

    map[NO_COMPRESSION] = shared_ptr<DeviceIntegerCODEC>(nullptr);
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

const map<COMPRESSION_TYPE, shared_ptr<DeviceIntegerCODEC>> DeviceCodecFactory::codecInstancesMap = initCodecInstancesMap();

// NOTE: Template instantiation of match_integrated is in match_integrated.cu
//
// TODO: Figure out a way how to separate match_integrated instances into multiple files, similar to how Codecs
// templates are instantiated in their respective .cu files

map<COMPRESSION_TYPE, MatchIntegratedFunPtr> initIntegratedKernelsMap()
{
    map<COMPRESSION_TYPE, MatchIntegratedFunPtr> map;

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

const map<COMPRESSION_TYPE, MatchIntegratedFunPtr> DeviceCodecFactory::integratedKernelsMap = initIntegratedKernelsMap();

map<COMPRESSION_TYPE, string> initCompressionNamesMap()
{
    map<COMPRESSION_TYPE, string> map;    
    for (auto it = DeviceCodecFactory::codecInstancesMap.begin(); it != DeviceCodecFactory::codecInstancesMap.end(); it++)
    {
        if (it->second.get())
            map[it->first] = it->second->name();
    }
    map[NO_COMPRESSION] = string("no");
    return map;
}

const map<COMPRESSION_TYPE, string> DeviceCodecFactory::compressionNamesMap = initCompressionNamesMap();

map<string, COMPRESSION_TYPE> initCompressionTypesMap()
{
    map<string, COMPRESSION_TYPE> map;    
    for (auto it = DeviceCodecFactory::compressionNamesMap.begin(); it != DeviceCodecFactory::compressionNamesMap.end(); it++)
        map[it->second] = it->first;
    return map;
}

const map<string, COMPRESSION_TYPE> DeviceCodecFactory::compressionTypesMap = initCompressionTypesMap();

vector<string> initAllCompressionNames()
{
    vector<string> names;
    for (auto i = DeviceCodecFactory::compressionNamesMap.begin(); i != DeviceCodecFactory::compressionNamesMap.end(); ++i)
        names.push_back(i->second);
    return names;
}

const vector<string> DeviceCodecFactory::allCompressionNames = initAllCompressionNames();

vector<COMPRESSION_TYPE> initAllCompressionTypes()
{
    vector<COMPRESSION_TYPE> types;
    for (auto i = DeviceCodecFactory::integratedKernelsMap.begin(); i != DeviceCodecFactory::integratedKernelsMap.end(); ++i)
        types.push_back(i->first);
    return types;
}

const vector<COMPRESSION_TYPE> DeviceCodecFactory::allCompressionTypes = initAllCompressionTypes();

const shared_ptr<DeviceIntegerCODEC> DeviceCodecFactory::nullCodec = shared_ptr<DeviceIntegerCODEC>(nullptr);

