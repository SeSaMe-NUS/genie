#ifndef MT19937AR_H
#define MT19937AR_H

/* 
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)  
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

#include <float.h>
#include <stdio.h>

/* Period parameters */  
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

#define ENABLE_MT199937AR_FAST_RANDOM_NUMBERS_GENERATION 1

static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
extern "C" {
void init_genrand(unsigned long s)
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) {
        mt[mti] = 
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned long init_key[], int key_length)
{
    int i, j, k;
    init_genrand(19650218UL);
    i=1; j=0;
    k = (N>key_length ? N : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=N) { mt[0] = mt[N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=N-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=N) { mt[0] = mt[N-1]; i=1; }
    }

    mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
    unsigned long y;
    unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL); /* a default initial seed is used */

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }
  
    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/* generates a random number on [0,0x7fffffff]-interval */
long genrand_int31(void)
{
    return (long)(genrand_int32()>>1);
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(void)
{
    return genrand_int32()*(1.0/4294967295.0); 
    /* divided by 2^32-1 */ 
}

/* generates a random number on [0,1)-real-interval */
double genrand_real2(void)
{
    return genrand_int32()*(1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval */
double genrand_real3(void)
{
    return (((double)genrand_int32()) + 0.5)*(1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on [0,1) with 53-bit resolution*/
double genrand_res53(void) 
{ 
    unsigned long a=genrand_int32()>>5, b=genrand_int32()>>6; 
    return(a*67108864.0+b)*(1.0/9007199254740992.0); 
} 
/* These real versions are due to Isaku Wada, 2002/01/09 added */

/*
int main(void)
{
    int i;
    unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    init_by_array(init, length);
    printf("1000 outputs of genrand_int32()\n");
    for (i=0; i<1000; i++) {
      printf("%10lu ", genrand_int32());
      if (i%5==4) printf("\n");
    }
    printf("\n1000 outputs of genrand_real2()\n");
    for (i=0; i<1000; i++) {
      printf("%10.8f ", genrand_real2());
      if (i%5==4) printf("\n");
    }
    return 0;
}
*/
}

#include <set>
#include <math.h>
#include <cstdio>
#include <limits.h>
#include <stdlib.h>

#ifdef __linux__
#include <float.h>
#endif

template <typename T>
class OperatorAdd
{
public:
    T operator()(const T& a, const T& b) 
    { 
        return a + b;
    }

    T identity() { return (T)0; }
};

template <typename T>
class OperatorMultiply
{
public:
    T operator()(const T& a, const T& b) 
    { 
        return a * b; 
    }
    T identity() { return (T)1; }
};

template <typename T>
class OperatorMax
{
public:
    T operator() (const T& a, const T& b) const 
    { 
        return max(a, b); 
    }
    T identity() const { return (T)0; }
};

template <typename T>
class OperatorMin
{
public:
    T operator() (const T& a, const T& b) const 
    { 
        return min(a, b); 
    }
    T identity() const { return (T)0; }
};

template <>
int OperatorMax<int>::identity() const { return INT_MIN; }
template <>
unsigned int OperatorMax<unsigned int>::identity() const { return 0; }
template <>
float OperatorMax<float>::identity() const { return -FLT_MAX; }
template <>
double OperatorMax<double>::identity() const { return -DBL_MAX; }

template <>
int OperatorMin<int>::identity() const { return INT_MAX; }
template <>
unsigned int OperatorMin<unsigned int>::identity() const { return UINT_MAX; }
template <>
float OperatorMin<float>::identity() const { return FLT_MAX; }
template <>
double OperatorMin<double>::identity() const { return DBL_MAX; }

template <class T>
inline T getMax() { return 0; }

// type specializations for the above
// getMax
template <> inline int getMax() { return INT_MAX; }
template <> inline unsigned int getMax() { return UINT_MAX; }
template <> inline float getMax() { return FLT_MAX; }
template <> inline double getMax() { return DBL_MAX; }

// TODO - Should I make this a namespace
template <typename T>
class VectorSupport
{
public:
    static void fillVector(T *a, size_t numElements, 
                           unsigned int keybits, T range) {};
    static void fillVectorForHash(unsigned int *kI, 
                                  float        fracOfKeysNonExistent,
                                  float        avgRepeatPerKey,
                                  bool         repeatedKeys,
                                  bool         isSet,
                                  size_t       numElements,
                                  unsigned int seed = 95123) {};
};

template <>
class VectorSupport <int>
{
public:
    static void fillVector(int *a, size_t numElements, 
                           unsigned int keybits, int range);
    static void fillVectorForHash(unsigned int *kI, 
                                  float        fracOfKeysNonExistent,
                                  float        avgRepeatPerKey,
                                  bool         repeatedKeys,
                                  bool         isSet,
                                  size_t       numElements,
                                  unsigned int seed = 95123) ;
};

template <>
class VectorSupport <unsigned int>
{
public:
    static void fillVector(unsigned int *a, size_t numElements, 
                           unsigned int keybits, unsigned int range);
    static void fillVectorForHash(unsigned int *kI, 
                                  float        fracOfKeysNonExistent,
                                  float        avgRepeatPerKey,
                                  bool         repeatedKeys,
                                  bool         isSet,
                                  size_t       numElements,
                                  unsigned int seed = 95123);
};

template <>
class VectorSupport <float>
{
public:
    static void fillVector(float *a, size_t numElements, 
                           unsigned int keybits, float range);
    static void fillVectorForHash(unsigned int *kI,
                                  float        fracOfKeysNonExistent,
                                  float        avgRepeatPerKey,
                                  bool         repeatedKeys,
                                  bool         isSet,
                                  size_t       numElements,
                                  unsigned int seed = 95123);
};

template <>
class VectorSupport <double>
{
public:
    static void fillVector(double *a, size_t numElements, 
                           unsigned int keybits, double range);
    static void fillVectorForHash(unsigned int *kI, 
                                  float        fracOfKeysNonExistent,
                                  float        avgRepeatPerKey,
                                  bool         repeatedKeys,
                                  bool         isSet,
                                  size_t       numElements,
                                  unsigned int seed = 95123);
};

void 
VectorSupport<unsigned int>::fillVector(unsigned int *a,
                                        size_t       numElements, 
                                        unsigned int keybits, 
                                        unsigned int range)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;            

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
    }
}

void 
VectorSupport<int>::fillVector(int          *a,
                               size_t       numElements, 
                               unsigned int keybits, 
                               int          range)
{
    VectorSupport<unsigned int>::fillVector((unsigned int*)a, numElements,
                                             keybits, (unsigned int)range);
}

void
VectorSupport<float>::fillVector(float        *a, 
                                 size_t       numElements, 
                                 unsigned int keybits, 
                                 float        range)
{
    srand(95123);
    for(unsigned int j = 0; j < numElements; j++)
    {
        //a[j] = pow(-1,(float)j)*(float)((rand()<<16) | rand());          
        a[j] = pow(-1, float(j)) * (range * (rand() / float(RAND_MAX)));
    }
}

void 
VectorSupport<double>::fillVector(double       *a, 
                                  size_t       numElements, 
                                  unsigned int keybits, 
                                  double       range)
{
    srand(95123);
    for(unsigned int j = 0; j < numElements; j++)
    {
        a[j] = pow(-1, double(j)) * 
                       (range * (rand() / double(RAND_MAX)));
    }
}

void 
VectorSupport<unsigned int>::fillVectorForHash(unsigned int *kI,
                                               float        fracOfKeysNonExistent,
                                               float        avgRepeatPerKey,
                                               bool         repeatedKeys,
                                               bool         isSet,
                                               size_t       numElements,
                                               unsigned int seed)
{
    // If seed is different from the original default seed value
    if (seed != 95123)
    {
      mti=N+1;
    }

    // Use the lower 30 bits
    const unsigned int numBitsToUse = 30;
    const unsigned int numBitsInUnsignedInt = 
        sizeof(unsigned int) * 8;

    unsigned int extraElements = 
        (unsigned int)(floorf(fracOfKeysNonExistent * numElements));
   
    unsigned int *host_buffer = 
    (unsigned int*) 
        malloc ((numElements + extraElements) * sizeof(unsigned int));

    // Seed the Random Number Generator
    init_genrand(seed);

    // Generate a random set of keys through the universe.
#if (!ENABLE_MT199937AR_FAST_RANDOM_NUMBERS_GENERATION)
    std::set<unsigned int> generatedKeys;
    for (size_t i = 0; i < (numElements + extraElements); i++) 
    {
        unsigned int newKey;
        do 
        {
            newKey = (genrand_int32() >> 
                     (numBitsInUnsignedInt - numBitsToUse));
        } while (generatedKeys.find(newKey) != generatedKeys.end());
        generatedKeys.insert(newKey);
  
        host_buffer[i]  = newKey;
    }
#else
    unsigned int max_range = (0xFFFFFFFFu) - 1u;
    unsigned int size      = numElements;
    int offset_range       = max_range / (size + 1);
    // Avoid power of two offset_range, to avoid aligned patterns in the generated random values
    offset_range     = (!(offset_range & (offset_range - 1)) && offset_range) ? offset_range - 1 : offset_range;
    offset_range     = max(1, offset_range);
    for (size_t i = 0; i < (numElements + extraElements); i++) 
    {
        unsigned int newKey;
        newKey = (genrand_int32() >> (numBitsInUnsignedInt - numBitsToUse));

        unsigned int o = newKey % offset_range;
        newKey = ((i * offset_range) + o);

        host_buffer[i]  = newKey;
    }
#endif


    // Copy the first numElements keys into h_keysInsert
    for (size_t i = 0; i < numElements; ++i)
    {
        kI[i] = host_buffer[i];
    }

    free(host_buffer);
}

#endif