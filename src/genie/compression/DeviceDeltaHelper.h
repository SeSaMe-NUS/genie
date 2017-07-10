/**
 * This code is a modification of code from SIMDIntersectionAndCompression library by Leonid Boytsov, Nathan Kurz and
 * Daniel Lemire
 */

#ifndef DEVICE_DELTA_HELPER_H_
#define DEVICE_DELTA_HELPER_H_

namespace genie
{
namespace compression
{

template <class T>
struct DeviceDeltaHelper {

    static void
    delta(const T initoffset, T *data, const size_t size) {
        if (size == 0)
            return; // nothing to do
        if (size > 1)
            for (size_t i = size - 1; i > 0; --i) {
                data[i] -= data[i - 1];
            }
        data[0] -= initoffset;
    }

    static void
    inverseDelta(const T initoffset, T *data, const size_t size) {
        if (size == 0)
            return; // nothing to do
        data[0] += initoffset;
        const size_t UnrollQty = 4;
        const size_t sz0 =
            (size / UnrollQty) * UnrollQty; // equal to 0, if size < UnrollQty
        size_t i = 1;
        if (sz0 >= UnrollQty) {
            T a = data[0];
            for (; i < sz0 - UnrollQty; i += UnrollQty) {
                a = data[i] += a;
                a = data[i + 1] += a;
                a = data[i + 2] += a;
                a = data[i + 3] += a;
            }
        }
        for (; i != size; ++i) {
            data[i] += data[i - 1];
        }
    }

    __device__ static void
    inverseDeltaOnGPU(const T initoffset, T *d_data, const size_t size) {
        if (size == 0)
            return; // nothing to do
        d_data[0] += initoffset;
        for (size_t i = 1; i != size; ++i) {
            d_data[i] += d_data[i - 1];
        }
    }
};

} // namespace compression
} // namespace genie

#endif
