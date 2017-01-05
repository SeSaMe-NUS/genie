#ifndef DEVICE_COMPOSITE_CODEC_H_
#define DEVICE_COMPOSITE_CODEC_H_

#include <SIMDCAI/common.h>
#include <SIMDCAI/util.h>
#include <SIMDCAI/codecs.h>

#include "DeviceDeltaHelper.h"

namespace GPUGenie {


class DeviceIntegerCODEC : public IntegerCODEC {

	virtual void
	encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) = 0;

	virtual const uint32_t*
	decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) = 0;

	__device__ virtual const uint32_t*
	decodeArrayOnGPU(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) = 0;

	virtual
	~DeviceIntegerCODEC() {}

	/** Convenience function not supported */
	virtual vector<uint32_t>
	compress(vector<uint32_t> &data) {
		throw std::logic_error("DeviceIntegerCODEC::compress not supported!");
	}

	/** Convenience function not supported */
	virtual vector<uint32_t>
	uncompress(vector<uint32_t> &compresseddata, size_t expected_uncompressed_size = 0) {
		throw std::logic_error("DeviceIntegerCODEC::uncompress not supported!");
	}

	virtual string
	name() const = 0;
};

class DeviceJustCopyCodec : public DeviceIntegerCODEC {

	virtual void
	encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
	{
		std::memcpy(out, in, sizeof(uint32_t) * length);
		nvalue = length;
	}

	virtual const uint32_t*
	decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
	{
		std::memcpy(out, in, sizeof(uint32_t) * length);
		nvalue = length;
		return in + length;
	}

	__device__ virtual const uint32_t*
	decodeArrayOnGPU(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue)
	{
		cudaCheckErrors(cudaMemCpy(d_out, d_in, sizeof(uint32_t) * length, cudaDeviceToDevice));
		nvalue = length;
		return d_in + length;
	}

	virtual
	~DeviceIntegerCODEC() {}

	virtual string
	name() const { return "DeviceJustCopyCodec"; }
};


class DeviceDeltaCodec : public DeviceIntegerCODEC {

	virtual void
	encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
	{
		std::memcpy(out, in, sizeof(uint32_t) * length);
		DeviceDeltaHelper<uint32_t>::delta(0 , out, const size_t length) 
		nvalue = length;
	}

	virtual const uint32_t*
	decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
	{
		std::memcpy(out, in, sizeof(uint32_t) * length);
		DeviceDeltaHelper<uint32_t>::inversedelta(0 , out, const size_t length) 
		nvalue = length;
		return in + length;
	}

	__device__ virtual const uint32_t*
	decodeArrayOnGPU(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue)
	{
		cudaCheckErrors(cudaMemCpy(d_out, d_in, sizeof(uint32_t) * length, cudaDeviceToDevice));
		DeviceDeltaHelper<uint32_t>::inversedeltaOnGPU(0 , out, const size_t length) 
		nvalue = length;
		return d_in + length;
	}

	virtual
	~DeviceIntegerCODEC() {}

	virtual string
	name() const { return "DeviceDeltaCodec"; }
};



/**
 * Same as SIMDCAI::CompositeCodes, but support decoding on GPU
 *
 * This is a useful class for CODEC that only compress data having length a multiple of some unit length.
 */
template <class Codec1, class Codec2>
class DeviceCompositeCodec : public DeviceIntegerCODEC {
public:

	DeviceCompositeCodec() : codec1(), codec2() {}
	Codec1 codec1;
	Codec2 codec2;

	virtual void
	encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
		const size_t roundedlength = length / Codec1::BlockSize * Codec1::BlockSize;
		size_t nvalue1 = nvalue;
		codec1.encodeArray(in, roundedlength, out, nvalue1);

		if (roundedlength < length) {
			ASSERT(nvalue >= nvalue1, nvalue << " " << nvalue1);
			size_t nvalue2 = nvalue - nvalue1;
			codec2.encodeArray(in + roundedlength, length - roundedlength, out + nvalue1, nvalue2);
			nvalue = nvalue1 + nvalue2;
		} else {
			nvalue = nvalue1;
		}
	}

	virtual const uint32_t*
	decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
		const uint32_t *const initin(in);
		size_t mynvalue1 = nvalue;
		const uint32_t *in2 = codec1.decodeArray(in, length, out, mynvalue1);
		if (length + in > in2) {
			assert(nvalue > mynvalue1);
			size_t nvalue2 = nvalue - mynvalue1;
			const uint32_t *in3 = codec2.decodeArray(in2, length - (in2 - in), out + mynvalue1, nvalue2);
			nvalue = mynvalue1 + nvalue2;
			assert(initin + length >= in3);
			return in3;
		}
		nvalue = mynvalue1;
		assert(initin + length >= in2);
		return in2;
	}

	__device__ virtual const uint32_t*
	decodeArrayOnGPU(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) {
		const uint32_t *const initin(in);
		size_t mynvalue1 = nvalue;
		const uint32_t *in2 = codec1.decodeArrayOnGPU(in, length, out, mynvalue1);
		if (length + in > in2) {
			assert(nvalue > mynvalue1);
			size_t nvalue2 = nvalue - mynvalue1;
			const uint32_t *in3 = codec2.decodeArrayOnGPu(in2, length - (in2 - in), out + mynvalue1, nvalue2);
			nvalue = mynvalue1 + nvalue2;
			assert(initin + length >= in3);
			return in3;
		}
		nvalue = mynvalue1;
		assert(initin + length >= in2);
		return in2;
	}

	string name() const {
		ostringstream convert;
		convert << "DeviceCompositeCodec_" codec1.name() << "+" << codec2.name();
		return convert.str();
	}
};

}

#endif
