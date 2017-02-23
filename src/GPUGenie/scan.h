/*
 * This module contains source code provided by NVIDIA Corporation.
 */

#ifndef SCAN_H
#define SCAN_H

#include <stdlib.h>

extern const unsigned int MIN_SHORT_ARRAY_SIZE;
extern const unsigned int MAX_SHORT_ARRAY_SIZE;
extern const unsigned int MIN_LARGE_ARRAY_SIZE;
extern const unsigned int MAX_LARGE_ARRAY_SIZE;

void initScan(void);
void closeScan(void);

size_t scanExclusiveShort(
    unsigned int *d_Dst,
    unsigned int *d_Src,
    unsigned int arrayLength);

size_t scanExclusiveLarge(
    unsigned int *d_Dst,
    unsigned int *d_Src,
    unsigned int arrayLength);

void scanExclusiveHost(
    unsigned int *dst,
    unsigned int *src,
    unsigned int arrayLength);

#endif
