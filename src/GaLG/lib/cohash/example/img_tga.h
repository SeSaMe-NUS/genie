/*
 *  (C) copyright  2011, Ismael Garcia, (U.Girona/ViRVIG, Spain & INRIA/ALICE, France)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef __TGA_H__
#define __TGA_H__

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
 
#define uchar unsigned char
#define sint short int
 
typedef struct
{
        uchar depth;
        sint w, h;
        uchar* data;
} Texture;
 
Texture* loadTGA(char* fn);

void saveTGA(Texture *tga, char *filename);

#endif






