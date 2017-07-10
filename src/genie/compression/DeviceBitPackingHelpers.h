/**
 * This code is a modification of code from SIMDIntersectionAndCompression library by Leonid Boytsov, Nathan Kurz and
 * Daniel Lemire
 */

#ifndef DEVICE_BIT_PACKING_HELPERS_H_
#define DEVICE_BIT_PACKING_HELPERS_H_

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "DeviceDeltaHelper.h"

namespace genie
{
namespace compression
{

void __device__ __host__ __fastunpack0 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack1 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack2 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack3 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack4 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack5 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack6 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack7 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack8 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack9 (const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack10(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack11(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack12(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack13(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack14(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack15(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack16(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack17(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack18(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack19(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack20(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack21(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack22(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack23(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack24(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack25(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack26(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack27(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack28(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack29(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack30(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack31(const uint32_t * in, uint32_t * out);
void __device__ __host__ __fastunpack32(const uint32_t * in, uint32_t * out);

void __fastpack0 (const uint32_t * in, uint32_t * out);
void __fastpack1 (const uint32_t * in, uint32_t * out);
void __fastpack2 (const uint32_t * in, uint32_t * out);
void __fastpack3 (const uint32_t * in, uint32_t * out);
void __fastpack4 (const uint32_t * in, uint32_t * out);
void __fastpack5 (const uint32_t * in, uint32_t * out);
void __fastpack6 (const uint32_t * in, uint32_t * out);
void __fastpack7 (const uint32_t * in, uint32_t * out);
void __fastpack8 (const uint32_t * in, uint32_t * out);
void __fastpack9 (const uint32_t * in, uint32_t * out);
void __fastpack10(const uint32_t * in, uint32_t * out);
void __fastpack11(const uint32_t * in, uint32_t * out);
void __fastpack12(const uint32_t * in, uint32_t * out);
void __fastpack13(const uint32_t * in, uint32_t * out);
void __fastpack14(const uint32_t * in, uint32_t * out);
void __fastpack15(const uint32_t * in, uint32_t * out);
void __fastpack16(const uint32_t * in, uint32_t * out);
void __fastpack17(const uint32_t * in, uint32_t * out);
void __fastpack18(const uint32_t * in, uint32_t * out);
void __fastpack19(const uint32_t * in, uint32_t * out);
void __fastpack20(const uint32_t * in, uint32_t * out);
void __fastpack21(const uint32_t * in, uint32_t * out);
void __fastpack22(const uint32_t * in, uint32_t * out);
void __fastpack23(const uint32_t * in, uint32_t * out);
void __fastpack24(const uint32_t * in, uint32_t * out);
void __fastpack25(const uint32_t * in, uint32_t * out);
void __fastpack26(const uint32_t * in, uint32_t * out);
void __fastpack27(const uint32_t * in, uint32_t * out);
void __fastpack28(const uint32_t * in, uint32_t * out);
void __fastpack29(const uint32_t * in, uint32_t * out);
void __fastpack30(const uint32_t * in, uint32_t * out);
void __fastpack31(const uint32_t * in, uint32_t * out);
void __fastpack32(const uint32_t * in, uint32_t * out);

void __fastpackwithoutmask0 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask1 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask2 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask3 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask4 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask5 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask6 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask7 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask8 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask9 (const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask10(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask11(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask12(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask13(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask14(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask15(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask16(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask17(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask18(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask19(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask20(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask21(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask22(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask23(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask24(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask25(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask26(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask27(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask28(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask29(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask30(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask31(const uint32_t * in, uint32_t * out);
void __fastpackwithoutmask32(const uint32_t * in, uint32_t * out);

void __device__ __host__ __integratedfastunpack0(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack1(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack2(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack3(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack4(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack5(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack6(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack7(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack8(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack9(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack10(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack11(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack12(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack13(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack14(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack15(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack16(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack17(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack18(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack19(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack20(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack21(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack22(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack23(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack24(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack25(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack26(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack27(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack28(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack29(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack30(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack31(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __device__ __host__ __integratedfastunpack32(const uint32_t initoffset, const uint32_t * in, uint32_t * out);

void __integratedfastpack0 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack1 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack2 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack3 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack4 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack5 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack6 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack7 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack8 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack9 (const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack10(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack11(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack12(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack13(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack14(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack15(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack16(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack17(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack18(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack19(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack20(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack21(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack22(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack23(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack24(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack25(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack26(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack27(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack28(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack29(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack30(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack31(const uint32_t initoffset, const uint32_t * in, uint32_t * out);
void __integratedfastpack32(const uint32_t initoffset, const uint32_t * in, uint32_t * out);


struct DeviceBitPackingHelpers {
    const static unsigned BlockSize = 32;

    __device__ __host__ static void inline
    fastunpack(const uint32_t * in, uint32_t * out, const uint32_t bit)
    {
        // Could have used function pointers instead of switch.
        // Switch calls do offer the compiler more opportunities for optimization in
        // theory. In this case, it makes no difference with a good compiler.
        switch (bit) {
        case 0:
            __fastunpack0(in, out);
            break;
        case 1:
            __fastunpack1(in, out);
            break;
        case 2:
            __fastunpack2(in, out);
            break;
        case 3:
            __fastunpack3(in, out);
            break;
        case 4:
            __fastunpack4(in, out);
            break;
        case 5:
            __fastunpack5(in, out);
            break;
        case 6:
            __fastunpack6(in, out);
            break;
        case 7:
            __fastunpack7(in, out);
            break;
        case 8:
            __fastunpack8(in, out);
            break;
        case 9:
            __fastunpack9(in, out);
            break;
        case 10:
            __fastunpack10(in, out);
            break;
        case 11:
            __fastunpack11(in, out);
            break;
        case 12:
            __fastunpack12(in, out);
            break;
        case 13:
            __fastunpack13(in, out);
            break;
        case 14:
            __fastunpack14(in, out);
            break;
        case 15:
            __fastunpack15(in, out);
            break;
        case 16:
            __fastunpack16(in, out);
            break;
        case 17:
            __fastunpack17(in, out);
            break;
        case 18:
            __fastunpack18(in, out);
            break;
        case 19:
            __fastunpack19(in, out);
            break;
        case 20:
            __fastunpack20(in, out);
            break;
        case 21:
            __fastunpack21(in, out);
            break;
        case 22:
            __fastunpack22(in, out);
            break;
        case 23:
            __fastunpack23(in, out);
            break;
        case 24:
            __fastunpack24(in, out);
            break;
        case 25:
            __fastunpack25(in, out);
            break;
        case 26:
            __fastunpack26(in, out);
            break;
        case 27:
            __fastunpack27(in, out);
            break;
        case 28:
            __fastunpack28(in, out);
            break;
        case 29:
            __fastunpack29(in, out);
            break;
        case 30:
            __fastunpack30(in, out);
            break;
        case 31:
            __fastunpack31(in, out);
            break;
        case 32:
            __fastunpack32(in, out);
            break;
        default:
            break;
        }
    }

    static void inline
    fastpack(const uint32_t * in, uint32_t * out, const uint32_t bit)
    {
        // Could have used function pointers instead of switch.
        // Switch calls do offer the compiler more opportunities for optimization in
        // theory. In this case, it makes no difference with a good compiler.
        switch (bit) {
        case 0:
            __fastpack0(in, out);
            break;
        case 1:
            __fastpack1(in, out);
            break;
        case 2:
            __fastpack2(in, out);
            break;
        case 3:
            __fastpack3(in, out);
            break;
        case 4:
            __fastpack4(in, out);
            break;
        case 5:
            __fastpack5(in, out);
            break;
        case 6:
            __fastpack6(in, out);
            break;
        case 7:
            __fastpack7(in, out);
            break;
        case 8:
            __fastpack8(in, out);
            break;
        case 9:
            __fastpack9(in, out);
            break;
        case 10:
            __fastpack10(in, out);
            break;
        case 11:
            __fastpack11(in, out);
            break;
        case 12:
            __fastpack12(in, out);
            break;
        case 13:
            __fastpack13(in, out);
            break;
        case 14:
            __fastpack14(in, out);
            break;
        case 15:
            __fastpack15(in, out);
            break;
        case 16:
            __fastpack16(in, out);
            break;
        case 17:
            __fastpack17(in, out);
            break;
        case 18:
            __fastpack18(in, out);
            break;
        case 19:
            __fastpack19(in, out);
            break;
        case 20:
            __fastpack20(in, out);
            break;
        case 21:
            __fastpack21(in, out);
            break;
        case 22:
            __fastpack22(in, out);
            break;
        case 23:
            __fastpack23(in, out);
            break;
        case 24:
            __fastpack24(in, out);
            break;
        case 25:
            __fastpack25(in, out);
            break;
        case 26:
            __fastpack26(in, out);
            break;
        case 27:
            __fastpack27(in, out);
            break;
        case 28:
            __fastpack28(in, out);
            break;
        case 29:
            __fastpack29(in, out);
            break;
        case 30:
            __fastpack30(in, out);
            break;
        case 31:
            __fastpack31(in, out);
            break;
        case 32:
            __fastpack32(in, out);
            break;
        default:
            break;
        }
    }

    /*assumes that integers fit in the prescribed number of bits*/
    static void inline
    fastpackwithoutmask(const uint32_t * in, uint32_t * out, const uint32_t bit) {
        // Could have used function pointers instead of switch.
        // Switch calls do offer the compiler more opportunities for optimization in
        // theory. In this case, it makes no difference with a good compiler.
        switch (bit) {
        case 0:
            __fastpackwithoutmask0(in, out);
            break;
        case 1:
            __fastpackwithoutmask1(in, out);
            break;
        case 2:
            __fastpackwithoutmask2(in, out);
            break;
        case 3:
            __fastpackwithoutmask3(in, out);
            break;
        case 4:
            __fastpackwithoutmask4(in, out);
            break;
        case 5:
            __fastpackwithoutmask5(in, out);
            break;
        case 6:
            __fastpackwithoutmask6(in, out);
            break;
        case 7:
            __fastpackwithoutmask7(in, out);
            break;
        case 8:
            __fastpackwithoutmask8(in, out);
            break;
        case 9:
            __fastpackwithoutmask9(in, out);
            break;
        case 10:
            __fastpackwithoutmask10(in, out);
            break;
        case 11:
            __fastpackwithoutmask11(in, out);
            break;
        case 12:
            __fastpackwithoutmask12(in, out);
            break;
        case 13:
            __fastpackwithoutmask13(in, out);
            break;
        case 14:
            __fastpackwithoutmask14(in, out);
            break;
        case 15:
            __fastpackwithoutmask15(in, out);
            break;
        case 16:
            __fastpackwithoutmask16(in, out);
            break;
        case 17:
            __fastpackwithoutmask17(in, out);
            break;
        case 18:
            __fastpackwithoutmask18(in, out);
            break;
        case 19:
            __fastpackwithoutmask19(in, out);
            break;
        case 20:
            __fastpackwithoutmask20(in, out);
            break;
        case 21:
            __fastpackwithoutmask21(in, out);
            break;
        case 22:
            __fastpackwithoutmask22(in, out);
            break;
        case 23:
            __fastpackwithoutmask23(in, out);
            break;
        case 24:
            __fastpackwithoutmask24(in, out);
            break;
        case 25:
            __fastpackwithoutmask25(in, out);
            break;
        case 26:
            __fastpackwithoutmask26(in, out);
            break;
        case 27:
            __fastpackwithoutmask27(in, out);
            break;
        case 28:
            __fastpackwithoutmask28(in, out);
            break;
        case 29:
            __fastpackwithoutmask29(in, out);
            break;
        case 30:
            __fastpackwithoutmask30(in, out);
            break;
        case 31:
            __fastpackwithoutmask31(in, out);
            break;
        case 32:
            __fastpackwithoutmask32(in, out);
            break;
        default:
            break;
        }
    }

    __device__ __host__ static void inline
    integratedfastunpack(const uint32_t initoffset, const uint32_t * in, uint32_t * out,
                         const uint32_t bit)
    {
        // Could have used function pointers instead of switch.
        // Switch calls do offer the compiler more opportunities for optimization in
        // theory. In this case, it makes no difference with a good compiler.
        switch (bit) {
        case 0:
            __integratedfastunpack0(initoffset, in, out);
            break;
        case 1:
            __integratedfastunpack1(initoffset, in, out);
            break;
        case 2:
            __integratedfastunpack2(initoffset, in, out);
            break;
        case 3:
            __integratedfastunpack3(initoffset, in, out);
            break;
        case 4:
            __integratedfastunpack4(initoffset, in, out);
            break;
        case 5:
            __integratedfastunpack5(initoffset, in, out);
            break;
        case 6:
            __integratedfastunpack6(initoffset, in, out);
            break;
        case 7:
            __integratedfastunpack7(initoffset, in, out);
            break;
        case 8:
            __integratedfastunpack8(initoffset, in, out);
            break;
        case 9:
            __integratedfastunpack9(initoffset, in, out);
            break;
        case 10:
            __integratedfastunpack10(initoffset, in, out);
            break;
        case 11:
            __integratedfastunpack11(initoffset, in, out);
            break;
        case 12:
            __integratedfastunpack12(initoffset, in, out);
            break;
        case 13:
            __integratedfastunpack13(initoffset, in, out);
            break;
        case 14:
            __integratedfastunpack14(initoffset, in, out);
            break;
        case 15:
            __integratedfastunpack15(initoffset, in, out);
            break;
        case 16:
            __integratedfastunpack16(initoffset, in, out);
            break;
        case 17:
            __integratedfastunpack17(initoffset, in, out);
            break;
        case 18:
            __integratedfastunpack18(initoffset, in, out);
            break;
        case 19:
            __integratedfastunpack19(initoffset, in, out);
            break;
        case 20:
            __integratedfastunpack20(initoffset, in, out);
            break;
        case 21:
            __integratedfastunpack21(initoffset, in, out);
            break;
        case 22:
            __integratedfastunpack22(initoffset, in, out);
            break;
        case 23:
            __integratedfastunpack23(initoffset, in, out);
            break;
        case 24:
            __integratedfastunpack24(initoffset, in, out);
            break;
        case 25:
            __integratedfastunpack25(initoffset, in, out);
            break;
        case 26:
            __integratedfastunpack26(initoffset, in, out);
            break;
        case 27:
            __integratedfastunpack27(initoffset, in, out);
            break;
        case 28:
            __integratedfastunpack28(initoffset, in, out);
            break;
        case 29:
            __integratedfastunpack29(initoffset, in, out);
            break;
        case 30:
            __integratedfastunpack30(initoffset, in, out);
            break;
        case 31:
            __integratedfastunpack31(initoffset, in, out);
            break;
        case 32:
            __integratedfastunpack32(initoffset, in, out);
            break;
        default:
            break;
        }
    }

    /*assumes that integers fit in the prescribed number of bits*/
    static void inline
    integratedfastpackwithoutmask(const uint32_t initoffset, const uint32_t * in,
                                  uint32_t * out, const uint32_t bit)
    {
        // Could have used function pointers instead of switch.
        // Switch calls do offer the compiler more opportunities for optimization in
        // theory. In this case, it makes no difference with a good compiler.
        switch (bit) {
        case 0:
            __integratedfastpack0(initoffset, in, out);
            break;
        case 1:
            __integratedfastpack1(initoffset, in, out);
            break;
        case 2:
            __integratedfastpack2(initoffset, in, out);
            break;
        case 3:
            __integratedfastpack3(initoffset, in, out);
            break;
        case 4:
            __integratedfastpack4(initoffset, in, out);
            break;
        case 5:
            __integratedfastpack5(initoffset, in, out);
            break;
        case 6:
            __integratedfastpack6(initoffset, in, out);
            break;
        case 7:
            __integratedfastpack7(initoffset, in, out);
            break;
        case 8:
            __integratedfastpack8(initoffset, in, out);
            break;
        case 9:
            __integratedfastpack9(initoffset, in, out);
            break;
        case 10:
            __integratedfastpack10(initoffset, in, out);
            break;
        case 11:
            __integratedfastpack11(initoffset, in, out);
            break;
        case 12:
            __integratedfastpack12(initoffset, in, out);
            break;
        case 13:
            __integratedfastpack13(initoffset, in, out);
            break;
        case 14:
            __integratedfastpack14(initoffset, in, out);
            break;
        case 15:
            __integratedfastpack15(initoffset, in, out);
            break;
        case 16:
            __integratedfastpack16(initoffset, in, out);
            break;
        case 17:
            __integratedfastpack17(initoffset, in, out);
            break;
        case 18:
            __integratedfastpack18(initoffset, in, out);
            break;
        case 19:
            __integratedfastpack19(initoffset, in, out);
            break;
        case 20:
            __integratedfastpack20(initoffset, in, out);
            break;
        case 21:
            __integratedfastpack21(initoffset, in, out);
            break;
        case 22:
            __integratedfastpack22(initoffset, in, out);
            break;
        case 23:
            __integratedfastpack23(initoffset, in, out);
            break;
        case 24:
            __integratedfastpack24(initoffset, in, out);
            break;
        case 25:
            __integratedfastpack25(initoffset, in, out);
            break;
        case 26:
            __integratedfastpack26(initoffset, in, out);
            break;
        case 27:
            __integratedfastpack27(initoffset, in, out);
            break;
        case 28:
            __integratedfastpack28(initoffset, in, out);
            break;
        case 29:
            __integratedfastpack29(initoffset, in, out);
            break;
        case 30:
            __integratedfastpack30(initoffset, in, out);
            break;
        case 31:
            __integratedfastpack31(initoffset, in, out);
            break;
        case 32:
            __integratedfastpack32(initoffset, in, out);
            break;
        default:
            break;
        }
    }

    template <class T> static
    void delta(const T initoffset, T *data, const size_t size) {
        if (size == 0)
            return; // nothing to do
        if (size > 1)
            for (size_t i = size - 1; i > 0; --i) {
                data[i] -= data[i - 1];
            }
        data[0] -= initoffset;
    }

    template <size_t size, class T>
    static void delta(const T initoffset, T *data) {
        if (size == 0)
            return; // nothing to do
        if (size > 1)
            for (size_t i = size - 1; i > 0; --i) {
                data[i] -= data[i - 1];
            }
        data[0] -= initoffset;
    }

    template <class T>
    static void inverseDelta(const T initoffset, T *data, const size_t size) {
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
    template <size_t size, class T>
    static void inverseDelta(const T initoffset, T *data) {
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

    static void inline ipackwithoutmask(const uint32_t *in, const size_t Qty,
                                        uint32_t *out, const uint32_t bit) {
        if (Qty % BlockSize) {
            throw std::logic_error("Incorrect # of entries.");
        }
        uint32_t initoffset = 0;

        for (size_t k = 0; k < Qty / BlockSize; ++k) {
            integratedfastpackwithoutmask(initoffset, in + k * BlockSize,
                                          out + k * bit, bit);
            initoffset = *(in + k * BlockSize + BlockSize - 1);
        }
    }

    static void inline pack(uint32_t *in, const size_t Qty, uint32_t *out,
                            const uint32_t bit) {
        if (Qty % BlockSize) {
            throw std::logic_error("Incorrect # of entries.");
        }
        uint32_t initoffset = 0;

        for (size_t k = 0; k < Qty / BlockSize; ++k) {
            const uint32_t nextoffset = *(in + k * BlockSize + BlockSize - 1);
            if (bit < 32)
                delta<BlockSize,uint32_t>(initoffset, in + k * BlockSize);
            fastpack(in + k * BlockSize, out + k * bit, bit);
            initoffset = nextoffset;
        }
    }

    static void inline packWithoutDelta(uint32_t *in, const size_t Qty,
                                        uint32_t *out, const uint32_t bit) {
        for (size_t k = 0; k < Qty / BlockSize; ++k) {
            fastpack(in + k * BlockSize, out + k * bit, bit);
        }
    }

    static void inline unpack(const uint32_t *in, const size_t Qty, uint32_t *out,
                              const uint32_t bit) {
        if (Qty % BlockSize) {
            throw std::logic_error("Incorrect # of entries.");
        }
        uint32_t initoffset = 0;

        for (size_t k = 0; k < Qty / BlockSize; ++k) {
            fastunpack(in + k * bit, out + k * BlockSize, bit);
            if (bit < 32)
                inverseDelta<BlockSize,uint32_t>(initoffset, out + k * BlockSize);
            initoffset = *(out + k * BlockSize + BlockSize - 1);
        }
    }

    static void inline unpackWithoutDelta(const uint32_t *in, const size_t Qty,
                                          uint32_t *out, const uint32_t bit) {
        for (size_t k = 0; k < Qty / BlockSize; ++k) {
            fastunpack(in + k * bit, out + k * BlockSize, bit);
        }
    }

    static void inline packwithoutmask(uint32_t *in, const size_t Qty,
                                       uint32_t *out, const uint32_t bit) {
        if (Qty % BlockSize) {
            throw std::logic_error("Incorrect # of entries.");
        }
        uint32_t initoffset = 0;

        for (size_t k = 0; k < Qty / BlockSize; ++k) {
            const uint32_t nextoffset = *(in + k * BlockSize + BlockSize - 1);
            if (bit < 32)
                delta<BlockSize,uint32_t>(initoffset, in + k * BlockSize);
            fastpackwithoutmask(in + k * BlockSize, out + k * bit, bit);
            initoffset = nextoffset;
        }
    }

    static void inline packwithoutmaskWithoutDelta(uint32_t *in, const size_t Qty,
            uint32_t *out,
            const uint32_t bit) {
        for (size_t k = 0; k < Qty / BlockSize; ++k) {
            fastpackwithoutmask(in + k * BlockSize, out + k * bit, bit);
        }
    }

    static void inline iunpack(const uint32_t *in, const size_t Qty,
                               uint32_t *out, const uint32_t bit) {
        if (Qty % BlockSize) {
            throw std::logic_error("Incorrect # of entries.");
        }

        uint32_t initoffset = 0;
        for (size_t k = 0; k < Qty / BlockSize; ++k) {
            integratedfastunpack(initoffset, in + k * bit, out + k * BlockSize, bit);
            initoffset = *(out + k * BlockSize + BlockSize - 1);
        }
    }

    static void CheckMaxDiff(const std::vector<uint32_t> &refdata, unsigned bit) {
        for (size_t i = 1; i < refdata.size(); ++i) {
            if (gccbits(refdata[i] - refdata[i - 1]) > bit)
                throw std::runtime_error("bug");
        }
    }

    static inline uint32_t gccbits(const uint32_t v) {
        return v == 0 ? 0 : 32 - __builtin_clz(v);
    }
};

} // namespace compression
} // namespace genie

#endif
